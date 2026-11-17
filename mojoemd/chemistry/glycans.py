"""GLYCAM topology construction, connectivity checks, and diagnostics."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import argparse
import json
import logging
import math
import sys

from openmm.app import ForceField, Modeller, PDBFile, Topology

from mojoemd.chemistry.metals import build_metal_aware_protonation_variants
from mojoemd.chemistry.ncaa import patch_ncaa_internal_bonds_from_ffxml, patch_ncaa_peptide_bonds
from mojoemd.chemistry.protein import prepare_protein


PDB_TO_GLYCAM_SUFFIXES = {
    "ABE": ("AE", "AF"),
    "ADA": ("OA", "OB"),
    "AHR": ("AA", "AB", "AD", "AU"),
    "ALL": ("NA", "NB"),
    "ALT": ("EA", "EB"),
    "ARA": ("AA", "AB", "AD", "AU"),
    "BAC": ("BC",),
    "BGC": ("GB",),
    "BMA": ("MB",),
    "FRU": ("CA", "CB", "CD", "CU"),
    "FUC": ("FA",),
    "FUL": ("FA",),
    "GAL": ("LA", "LB"),
    "GCS": ("YN", "YS"),
    "GCU": ("ZA", "ZB"),
    "GLA": ("LA", "LB"),
    "GLC": ("GA", "GB"),
    "GL0": ("KA", "KB"),
    "GUL": ("KA", "KB"),
    "GUP": ("KA", "KB"),
    "IDS": ("UA", "UB"),
    "LYX": ("DA", "DB", "DD", "DU"),
    "LXC": ("XA", "XB", "XD", "XU"),
    "MAL": ("GA",),
    "MAN": ("MA",),
    "NAG": ("YB",),
    "NDG": ("YB",),
    "NGA": ("VA", "VB"),
    "QUI": ("QA", "QB"),
    "PSI": ("PA", "PB", "PD", "PU"),
    "RAM": ("HA", "HB"),
    "RIB": ("RA", "RB", "RD", "RU"),
    "SGN": ("YB",),
    "SIA": ("SA", "SB"),
    "SOR": ("BA", "BB", "BD", "BU"),
    "TAG": ("JA", "JB", "JD", "JU"),
    "TAL": ("TA", "TB"),
    "TYV": ("TV", "Tv"),
    "XYP": ("XA", "XB", "XD", "XU"),
    "XYS": ("XA", "XB", "XD", "XU"),
    "XYL": ("XA", "XB", "XD", "XU"),
}


GLYCAN_RESNAMES = set(PDB_TO_GLYCAM_SUFFIXES) | {
    "API", "BEM", "COL", "DHA", "DIG", "GMH", "IDO", "KDN", "KDO", "LGU", "MAV", "MUR", "NEU",
    "OLI", "PAR",
}


PROTEIN_GLYCAN_LINKS = {
    ("ASN", "ND2", "NAG", "C1"),
    ("ASN", "ND2", "NDG", "C1"),
    ("ASN", "ND2", "BMA", "C1"),
}


GLYCAN_LINK_DISTANCE_CUTOFF = 1.90


GLYCAN_GLYCAN_DISTANCE_CUTOFF = 1.85


GLYCAN_RING_ATOM = "C1"


GLYCAN_ANOMERIC_ATOMS = {
    "SIA": "C2",
    "NEU": "C2",
    "KDN": "C2",
    "KDO": "C2",
}


def _glyco_guess_element(atom_name: str, element_field: str = "") -> str:
    if element_field.strip():
        return element_field.strip().upper()

    atom_name = atom_name.strip()
    if atom_name and atom_name[0].isdigit():
        atom_name = atom_name[1:]

    if len(atom_name) >= 2:
        first2 = atom_name[:2].upper()
        if first2 in {"CL", "BR", "NA", "MG", "FE", "ZN"}:
            return first2

    return atom_name[0].upper() if atom_name else ""


def _glyco_is_hydrogen_pdb_atom_line(line: str) -> bool:
    atom_name = line[12:16].strip()
    element = _glyco_guess_element(line[12:16], line[76:78] if len(line) >= 78 else "")
    return element == "H" or atom_name.startswith("H")


def _glyco_parse_pdb(pdb_file: str | Path) -> tuple[list[dict], list[dict]]:
    atoms: list[dict] = []
    links: list[dict] = []

    with open(pdb_file) as handle:
        for line in handle:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"}:
                atoms.append({
                    "record": record,
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip() or " ",
                    "resid": int(line[22:26]),
                    "icode": line[26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": _glyco_guess_element(line[12:16], line[76:78] if len(line) >= 78 else ""),
                })

            elif record == "LINK":
                try:
                    links.append({
                        "atom1": line[12:16].strip(),
                        "resname1": line[17:20].strip(),
                        "chain1": line[21].strip() or " ",
                        "resid1": int(line[22:26]),
                        "icode1": line[26].strip(),
                        "atom2": line[42:46].strip(),
                        "resname2": line[47:50].strip(),
                        "chain2": line[51].strip() or " ",
                        "resid2": int(line[52:56]),
                        "icode2": line[56].strip(),
                    })
                except Exception:
                    continue

    return atoms, links


def _glyco_parse_link_line(line: str) -> dict | None:
    try:
        return {
            "atom1": line[12:16].strip(),
            "resname1": line[17:20].strip(),
            "chain1": line[21].strip() or " ",
            "resid1": int(line[22:26]),
            "icode1": line[26].strip(),
            "atom2": line[42:46].strip(),
            "resname2": line[47:50].strip(),
            "chain2": line[51].strip() or " ",
            "resid2": int(line[52:56]),
            "icode2": line[56].strip(),
        }
    except Exception:
        return None


def _glyco_residue_key(atom: dict) -> tuple[str, int, str, str]:
    return atom["chain"], atom["resid"], atom["icode"], atom["resname"]


def _glyco_residue_label(key: tuple[str, int, str, str]) -> str:
    chain, resid, icode, resname = key
    return f"{resname}:{chain}:{resid}{icode}"


def _glyco_parse_residue_label(label: str) -> tuple[str, str, str]:
    return tuple(label.split(":", 2))  # type: ignore[return-value]


def _glyco_group_residues(atoms: list[dict]) -> dict[tuple[str, int, str, str], list[dict]]:
    residues: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for atom in atoms:
        residues[_glyco_residue_key(atom)].append(atom)
    return residues


def _glyco_get_atom(residue_atoms: list[dict], atom_name: str) -> dict | None:
    for atom in residue_atoms:
        if atom["name"] == atom_name:
            return atom
    return None


def _glyco_distance(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _glyco_anomeric_atom(resname: str) -> str:
    return GLYCAN_ANOMERIC_ATOMS.get(resname, GLYCAN_RING_ATOM)


def _glyco_is_glycosidic_atom_pair(atom1: dict, atom2: dict) -> bool:
    """Restrict inferred glycan links to anomeric-carbon/oxygen glycosidic bonds."""
    if atom1["name"] == _glyco_anomeric_atom(atom1["resname"]) and atom2["element"] == "O" and atom2["name"].startswith("O"):
        return True
    if atom2["name"] == _glyco_anomeric_atom(atom2["resname"]) and atom1["element"] == "O" and atom1["name"].startswith("O"):
        return True
    return False


def _glyco_is_supported_protein_glycan_pair(key1: tuple, atom1: str, key2: tuple, atom2: str) -> bool:
    resname1 = key1[3]
    resname2 = key2[3]

    for protein_res, protein_atom, glycan_res, glycan_atom in PROTEIN_GLYCAN_LINKS:
        if resname1 == protein_res and atom1 == protein_atom and resname2 == glycan_res and atom2 == glycan_atom:
            return True
        if resname2 == protein_res and atom2 == protein_atom and resname1 == glycan_res and atom1 == glycan_atom:
            return True

    return False


def _glyco_is_supported_link_record(key1: tuple, atom1: dict | None, key2: tuple, atom2: dict | None) -> bool:
    if atom1 is None or atom2 is None:
        return False

    key1_is_glycan = key1[3] in GLYCAN_RESNAMES
    key2_is_glycan = key2[3] in GLYCAN_RESNAMES

    if key1_is_glycan and key2_is_glycan:
        return _glyco_is_glycosidic_atom_pair(atom1, atom2)

    if key1_is_glycan != key2_is_glycan:
        return _glyco_is_supported_protein_glycan_pair(key1, atom1["name"], key2, atom2["name"])

    return False


def _glyco_detect_links_from_records(residues: dict, pdb_links: list[dict]) -> list[dict]:
    links = []

    for link in pdb_links:
        key1 = (link["chain1"], link["resid1"], link["icode1"], link["resname1"])
        key2 = (link["chain2"], link["resid2"], link["icode2"], link["resname2"])

        if key1 not in residues or key2 not in residues:
            continue
        if key1[3] not in GLYCAN_RESNAMES and key2[3] not in GLYCAN_RESNAMES:
            continue

        atom1 = _glyco_get_atom(residues[key1], link["atom1"])
        atom2 = _glyco_get_atom(residues[key2], link["atom2"])
        if not _glyco_is_supported_link_record(key1, atom1, key2, atom2):
            logging.info(
                "Ignoring unsupported glycan LINK record: "
                f"{_glyco_residue_label(key1)}:{link['atom1']} -- "
                f"{_glyco_residue_label(key2)}:{link['atom2']}"
            )
            continue
        dist = round(_glyco_distance(atom1, atom2), 3) if atom1 and atom2 else None

        links.append({
            "residue1": _glyco_residue_label(key1),
            "atom1": link["atom1"],
            "residue2": _glyco_residue_label(key2),
            "atom2": link["atom2"],
            "distance": dist,
            "source": "LINK",
        })

    return links


def write_pdb_with_supported_glycan_links_only(pdb_file: str | Path, out_pdb: Path) -> Path:
    atoms, _ = _glyco_parse_pdb(pdb_file)
    heavy_atoms = [atom for atom in atoms if atom["element"] != "H" and not atom["name"].startswith("H")]
    residues = _glyco_group_residues(heavy_atoms)
    atoms_by_serial = {atom["serial"]: atom for atom in heavy_atoms}
    skipped_links = 0
    skipped_conects = 0
    skipped_hydrogens = 0

    def keep_conect_pair(src_serial: int, target_serial: int) -> bool:
        nonlocal skipped_conects

        atom1 = atoms_by_serial.get(src_serial)
        atom2 = atoms_by_serial.get(target_serial)

        if atom1 is None or atom2 is None:
            skipped_conects += 1
            return False

        key1 = _glyco_residue_key(atom1)
        key2 = _glyco_residue_key(atom2)

        if key1 == key2:
            return True

        if key1[3] not in GLYCAN_RESNAMES and key2[3] not in GLYCAN_RESNAMES:
            return True

        if _glyco_is_supported_link_record(key1, atom1, key2, atom2):
            return True

        skipped_conects += 1
        logging.info(
            "Removing unsupported glycan CONECT before PDBFixer: "
            f"{_glyco_residue_label(key1)}:{atom1['name']} -- "
            f"{_glyco_residue_label(key2)}:{atom2['name']}"
        )
        return False

    with open(pdb_file) as handle, open(out_pdb, "w") as out:
        for line in handle:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"} and _glyco_is_hydrogen_pdb_atom_line(line):
                skipped_hydrogens += 1
                continue

            if record == "CONECT":
                try:
                    src_serial = int(line[6:11])
                    target_serials = [
                        int(line[i:i + 5])
                        for i in range(11, len(line), 5)
                        if line[i:i + 5].strip()
                    ]
                except ValueError:
                    out.write(line)
                    continue

                kept_targets = [
                    target_serial
                    for target_serial in target_serials
                    if keep_conect_pair(src_serial, target_serial)
                ]

                if kept_targets:
                    out.write(f"CONECT{src_serial:5d}" + "".join(f"{serial:5d}" for serial in kept_targets) + "\n")
                continue

            if record != "LINK":
                out.write(line)
                continue

            link = _glyco_parse_link_line(line)
            if link is None:
                out.write(line)
                continue

            key1 = (link["chain1"], link["resid1"], link["icode1"], link["resname1"])
            key2 = (link["chain2"], link["resid2"], link["icode2"], link["resname2"])

            if key1[3] not in GLYCAN_RESNAMES and key2[3] not in GLYCAN_RESNAMES:
                out.write(line)
                continue

            atom1 = _glyco_get_atom(residues.get(key1, []), link["atom1"])
            atom2 = _glyco_get_atom(residues.get(key2, []), link["atom2"])
            if _glyco_is_supported_link_record(key1, atom1, key2, atom2):
                out.write(line)
            else:
                skipped_links += 1
                logging.info(
                    "Removing unsupported glycan LINK before PDBFixer: "
                    f"{_glyco_residue_label(key1)}:{link['atom1']} -- "
                    f"{_glyco_residue_label(key2)}:{link['atom2']}"
                )

    logging.info(
        "Wrote PDBFixer input with "
        f"{skipped_links} unsupported glycan LINK record(s), "
        f"{skipped_conects} unsupported/invalid CONECT edge(s), and "
        f"{skipped_hydrogens} hydrogen atom record(s) removed: {out_pdb}"
    )
    return out_pdb


def _glyco_detect_protein_glycan_links(residues: dict) -> list[dict]:
    found = []
    residue_keys = list(residues.keys())

    for key1 in residue_keys:
        for key2 in residue_keys:
            if key1 == key2:
                continue

            resname1 = key1[3]
            resname2 = key2[3]

            for protein_res, protein_atom, glycan_res, glycan_atom in PROTEIN_GLYCAN_LINKS:
                if resname1 != protein_res or resname2 != glycan_res:
                    continue

                atom1 = _glyco_get_atom(residues[key1], protein_atom)
                atom2 = _glyco_get_atom(residues[key2], glycan_atom)
                if atom1 is None or atom2 is None:
                    continue

                d = _glyco_distance(atom1, atom2)
                if d <= GLYCAN_LINK_DISTANCE_CUTOFF:
                    found.append({
                        "residue1": _glyco_residue_label(key1),
                        "atom1": protein_atom,
                        "residue2": _glyco_residue_label(key2),
                        "atom2": glycan_atom,
                        "distance": round(d, 3),
                        "source": "distance",
                    })

    return found


def _glyco_detect_glycan_glycan_links(residues: dict) -> list[dict]:
    found = []
    glycan_keys = [key for key in residues if key[3] in GLYCAN_RESNAMES]

    for i, key1 in enumerate(glycan_keys):
        for key2 in glycan_keys[i + 1:]:
            best = None
            best_d = 999.0

            for atom1 in residues[key1]:
                for atom2 in residues[key2]:
                    if atom1["element"] == "H" or atom2["element"] == "H":
                        continue
                    if not _glyco_is_glycosidic_atom_pair(atom1, atom2):
                        continue
                    d = _glyco_distance(atom1, atom2)
                    if d < best_d:
                        best_d = d
                        best = atom1, atom2

            if best is None or best_d > GLYCAN_GLYCAN_DISTANCE_CUTOFF:
                continue

            atom1, atom2 = best
            found.append({
                "residue1": _glyco_residue_label(key1),
                "atom1": atom1["name"],
                "residue2": _glyco_residue_label(key2),
                "atom2": atom2["name"],
                "distance": round(best_d, 3),
                "source": "distance",
            })

    return found


def _glyco_unique_links(all_links: list[dict]) -> list[dict]:
    unique = []
    seen = set()

    for link in all_links:
        key = tuple(sorted([(link["residue1"], link["atom1"]), (link["residue2"], link["atom2"])]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(link)

    return unique


def _glyco_build_graph(links: list[dict]) -> tuple[dict[str, list[str]], dict[frozenset, dict]]:
    graph: dict[str, list[str]] = defaultdict(list)
    edge_data: dict[frozenset, dict] = {}

    for link in links:
        r1, r2 = link["residue1"], link["residue2"]
        graph[r1].append(r2)
        graph[r2].append(r1)
        edge_data[frozenset((r1, r2))] = link

    return graph, edge_data


def _glyco_connected_components(graph: dict[str, list[str]]) -> list[list[str]]:
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue
        queue = deque([node])
        component = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(sorted(component))

    return components


def _glyco_is_glycan_label(label: str) -> bool:
    resname, _, _ = _glyco_parse_residue_label(label)
    return resname in GLYCAN_RESNAMES


def _glyco_choose_anchor(component: list[str]) -> str:
    protein_nodes = [node for node in component if not _glyco_is_glycan_label(node)]
    return sorted(protein_nodes)[0] if protein_nodes else sorted(component)[0]


def _glyco_orient_link(parent: str, child: str, edge_data: dict[frozenset, dict]) -> dict:
    link = edge_data[frozenset((parent, child))]
    if link["residue1"] == parent and link["residue2"] == child:
        return {"parent": parent, "parent_atom": link["atom1"], "child": child, "child_atom": link["atom2"], "distance": link.get("distance"), "source": link["source"]}
    return {"parent": parent, "parent_atom": link["atom2"], "child": child, "child_atom": link["atom1"], "distance": link.get("distance"), "source": link["source"]}


def _glyco_build_oriented_tree(root: str, graph: dict[str, list[str]], edge_data: dict[frozenset, dict]) -> dict:
    visited = set()

    def walk(node: str, parent: str | None = None) -> dict:
        visited.add(node)
        children = []
        oriented_links = []
        for neighbor in sorted(graph[node]):
            if neighbor == parent or neighbor in visited:
                continue
            oriented_links.append(_glyco_orient_link(node, neighbor, edge_data))
            children.append(walk(neighbor, node))
        return {"residue": node, "children": children, "links_to_children": oriented_links}

    return walk(root)


def _glyco_component_records(components: list[list[str]], graph: dict[str, list[str]], edge_data: dict[frozenset, dict]) -> list[dict]:
    records = []
    for i, component in enumerate(components, 1):
        anchor = _glyco_choose_anchor(component)
        records.append({
            "component_id": i,
            "anchor": anchor,
            "nodes": component,
            "glycan_nodes": [n for n in component if _glyco_is_glycan_label(n)],
            "protein_nodes": [n for n in component if not _glyco_is_glycan_label(n)],
            "tree": _glyco_build_oriented_tree(anchor, graph, edge_data),
        })
    return records


def _glyco_template_external_atom_index(forcefield: ForceField) -> dict[str, dict[frozenset[str], list[str]]]:
    index: dict[str, dict[frozenset[str], list[str]]] = defaultdict(lambda: defaultdict(list))

    for template_name, template in forcefield._templates.items():
        if len(template_name) != 3:
            continue

        suffix = template_name[-2:]
        external_atoms = frozenset(template.atoms[i].name for i in template.externalBonds)
        if not external_atoms:
            continue
        if not any(atom in external_atoms for atom in ("C1", "C2", "O1")):
            continue

        index[suffix][external_atoms].append(template_name)

    return index


def _glyco_choose_template_from_index(
    resname: str,
    external_atoms: set[str],
    template_index: dict[str, dict[frozenset[str], list[str]]],
) -> tuple[str | None, str, str]:
    suffixes = PDB_TO_GLYCAM_SUFFIXES.get(resname, ())
    if not suffixes:
        return None, "unsupported", f"No PDB-to-GLYCAM residue-family mapping for {resname}"

    ext_key = frozenset(external_atoms)
    candidates = []
    for suffix in suffixes:
        candidates.extend(template_index.get(suffix, {}).get(ext_key, []))

    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        template = candidates[0]
        return template, "high", f"Matched GLYCAM template {template} by external atoms {sorted(external_atoms)}"
    if len(candidates) > 1:
        return None, "ambiguous", f"External atoms {sorted(external_atoms)} match multiple GLYCAM templates: {', '.join(candidates)}"

    return None, "unsupported", (
        f"No GLYCAM template for {resname} with external atoms {sorted(external_atoms)} "
        f"in family/families {', '.join(suffixes)}"
    )


def _glyco_choose_template_from_connectivity(
    resname: str,
    external_atoms: set[str],
    incoming_links: list[tuple],
) -> tuple[str | None, str, str] | None:
    """Resolve common ambiguous GLYCAM terminal templates from parent linkage motifs."""
    if resname == "GAL" and external_atoms == {"C1"}:
        for parent, parent_atom, child_atom, _link in incoming_links:
            parent_resname = parent.split(":")[0]
            if parent_resname in {"NAG", "YB", "0YB", "3YB", "4YB", "6YB", "UYB", "VYB", "WYB", "QYB"}:
                if parent_atom == "O4" and child_atom == "C1":
                    return (
                        "0LB",
                        "motif",
                        "Resolved terminal Gal-beta-1,4-GlcNAc motif from NAG O4 -> GAL C1 linkage",
                    )

    return None


def _glyco_assign_templates(
    links: list[dict],
    glycan_residues: set[str],
    template_index: dict[str, dict[frozenset[str], list[str]]],
) -> dict[str, dict]:
    external_atoms: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, list] = defaultdict(list)
    outgoing: dict[str, list] = defaultdict(list)

    for link in links:
        r1, a1 = link["residue1"], link["atom1"]
        r2, a2 = link["residue2"], link["atom2"]

        if r1 in glycan_residues:
            external_atoms[r1].add(a1)
        if r2 in glycan_residues:
            external_atoms[r2].add(a2)

        if a2 == "C1":
            parent, parent_atom, child, child_atom = r1, a1, r2, a2
        elif a1 == "C1":
            parent, parent_atom, child, child_atom = r2, a2, r1, a1
        else:
            continue

        incoming[child].append((parent, parent_atom, child_atom, link))
        outgoing[parent].append((child, parent_atom, child_atom, link))

    assignments: dict[str, dict] = {}

    for residue in sorted(glycan_residues):
        resname = residue.split(":")[0]
        ext = external_atoms.get(residue, set())
        template, confidence, reason = _glyco_choose_template_from_index(resname, ext, template_index)
        if template is None and confidence == "ambiguous":
            connectivity_choice = _glyco_choose_template_from_connectivity(resname, ext, incoming.get(residue, []))
            if connectivity_choice is not None:
                template, confidence, reason = connectivity_choice

        assignments[residue] = {
            "resname": resname,
            "template": template,
            "confidence": confidence,
            "reason": reason,
            "external_atoms": sorted(ext),
            "expected_external_atoms": sorted(ext) if template else [],
            "incoming_links": [
                {"parent": p, "parent_atom": pa, "child_atom": ca, "source": l.get("source"), "distance": l.get("distance")}
                for p, pa, ca, l in incoming.get(residue, [])
            ],
            "outgoing_links": [
                {"child": c, "parent_atom": pa, "child_atom": ca, "source": l.get("source"), "distance": l.get("distance")}
                for c, pa, ca, l in outgoing.get(residue, [])
            ],
            "substituted_positions": sorted(a for a in ext if a.startswith("O")),
        }

    return assignments


def _glyco_tree_to_ascii(tree: dict, edge_data: dict[frozenset, dict], template_map: dict[str, str], prefix: str = "", is_last: bool = True, parent: str | None = None) -> list[str]:
    label = tree["residue"]
    if label in template_map:
        _, chain, resid = _glyco_parse_residue_label(label)
        display = f"{template_map[label]}:{chain}:{resid}"
    else:
        display = label

    if parent is None:
        line = display
    else:
        link = _glyco_orient_link(parent, label, edge_data)
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}{display}  ({link['parent_atom']}-{link['child_atom']}, d={link['distance'] if link['distance'] is not None else '-'}, {link['source']})"

    lines = [line]
    child_prefix = prefix
    if parent is not None:
        child_prefix += "    " if is_last else "│   "

    children = tree["children"]
    for i, child in enumerate(children):
        lines.extend(_glyco_tree_to_ascii(child, edge_data, template_map, child_prefix, i == len(children) - 1, label))
    return lines


def _glyco_write_outputs(outdir: Path, glycan_residues: set[str], links: list[dict], component_records: list[dict], graph: dict[str, list[str]], glycam_assignments: dict[str, dict], edge_data: dict[frozenset, dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    data = {
        "glycan_residues": sorted(glycan_residues),
        "links": links,
        "glycam_assignments": glycam_assignments,
        "components": component_records,
        "graph": {k: sorted(v) for k, v in graph.items()},
    }

    with open(outdir / "glycans.json", "w") as out:
        json.dump(data, out, indent=2)

    with open(outdir / "glycans.tsv", "w") as out:
        out.write("residue1\tatom1\tresidue2\tatom2\tdistance\tsource\n")
        for link in links:
            out.write(f"{link['residue1']}\t{link['atom1']}\t{link['residue2']}\t{link['atom2']}\t{link.get('distance', '-')}\t{link['source']}\n")

    with open(outdir / "glycam_assignments.tsv", "w") as out:
        out.write("residue\tresname\ttemplate\tconfidence\treason\texternal_atoms\n")
        for residue, a in glycam_assignments.items():
            out.write(f"{residue}\t{a['resname']}\t{a['template']}\t{a['confidence']}\t{a['reason']}\t{','.join(a['external_atoms'])}\n")

    template_map = {residue: a["template"] for residue, a in glycam_assignments.items() if a.get("template")}
    with open(outdir / "glycan_trees.txt", "w") as out:
        for record in component_records:
            out.write(f"Component {record['component_id']} | anchor={record['anchor']} | glycan_residues={len(record['glycan_nodes'])}\n")
            for line in _glyco_tree_to_ascii(record["tree"], edge_data, template_map):
                out.write(line + "\n")
            out.write("\n")


def detect_glycans_for_prepare(
    pdb_file: str | Path,
    outdir: Path,
    forcefield: ForceField,
    use_link_records: bool = True,
    use_distance_detection: bool = True,
) -> dict:
    logging.info("Detecting covalently bound glycans for GLYCAM preparation...")
    atoms, pdb_links = _glyco_parse_pdb(pdb_file)
    residues = _glyco_group_residues(atoms)
    glycan_residues = {_glyco_residue_label(k) for k in residues if k[3] in GLYCAN_RESNAMES}

    all_links: list[dict] = []
    if use_link_records:
        all_links.extend(_glyco_detect_links_from_records(residues, pdb_links))
    if use_distance_detection:
        all_links.extend(_glyco_detect_protein_glycan_links(residues))
        all_links.extend(_glyco_detect_glycan_glycan_links(residues))

    links = _glyco_unique_links(all_links)
    graph, edge_data = _glyco_build_graph(links)
    components = _glyco_connected_components(graph)
    component_records = _glyco_component_records(components, graph, edge_data)
    template_index = _glyco_template_external_atom_index(forcefield)
    assignments = _glyco_assign_templates(links, glycan_residues, template_index)

    _glyco_write_outputs(outdir, glycan_residues, links, component_records, graph, assignments, edge_data)

    logging.info(f"Detected {len(glycan_residues)} glycan residues")
    logging.info(f"Detected {len(links)} covalent glycan links")
    logging.info(f"Detected {len(component_records)} glycan components")

    template_map = {residue: a["template"] for residue, a in assignments.items() if a.get("template")}
    for record in component_records:
        logging.info(f"Glycan {record['component_id']} | anchor={record['anchor']} | glycan_residues={len(record['glycan_nodes'])}")
        for line in _glyco_tree_to_ascii(record["tree"], edge_data, template_map):
            logging.info("  " + line)

    return {
        "glycan_residues": sorted(glycan_residues),
        "links": links,
        "glycam_assignments": assignments,
        "components": component_records,
        "graph": {k: sorted(v) for k, v in graph.items()},
    }


def find_glycosylated_asn_labels_from_glycan_data(glycans_data: dict) -> set[str]:
    labels = set()
    for link in glycans_data.get("links", []):
        r1, a1 = link["residue1"], link["atom1"]
        r2, a2 = link["residue2"], link["atom2"]
        if r1.startswith("ASN:") and a1 == "ND2" and r2.startswith("NAG:") and a2 == "C1":
            labels.add(r1)
        elif r2.startswith("ASN:") and a2 == "ND2" and r1.startswith("NAG:") and a1 == "C1":
            labels.add(r2)
    return labels


def rename_glycosylated_asn_to_nln(modeller: Modeller, glycans_data: dict) -> None:
    glyco_asn_labels = find_glycosylated_asn_labels_from_glycan_data(glycans_data)
    renamed = []
    for residue in modeller.topology.residues():
        label_as_asn = f"ASN:{residue.chain.id}:{residue.id}"
        if label_as_asn in glyco_asn_labels:
            old = residue.name
            residue.name = "NLN"
            renamed.append((label_as_asn, old, residue.name))
    for label, old, new in renamed:
        logging.info(f"Renamed glycosylated residue {label}: {old} -> {new}")
    logging.info(f"Renamed {len(renamed)} glycosylated ASN residues to NLN")


def rename_glycan_atoms_to_glycam(modeller: Modeller) -> None:
    atom_rename = {"NAG": {"C7": "C2N", "O7": "O2N", "C8": "CME"}}
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for atom in modeller.topology.atoms():
        residue = atom.residue
        mapping = atom_rename.get(residue.name)
        if mapping and atom.name in mapping:
            old = atom.name
            atom.name = mapping[old]
            grouped[f"{residue.name}:{residue.chain.id}:{residue.id}"].append((old, atom.name))
    for label, changes in grouped.items():
        logging.info(f"GLYCAM atom renaming for {label}: " + ", ".join(f"{o}->{n}" for o, n in changes))
    logging.info(f"Renamed {sum(len(v) for v in grouped.values())} glycan atoms to GLYCAM atom names")


def rename_glycan_residues_to_glycam_templates(modeller: Modeller, glycans_data: dict) -> None:
    assignments = glycans_data.get("glycam_assignments", {})
    renamed = []
    for residue in modeller.topology.residues():
        label = f"{residue.name}:{residue.chain.id}:{residue.id}"
        if label not in assignments:
            continue
        template = assignments[label].get("template")
        if not template:
            logging.warning(f"No GLYCAM template assignment for {label}; leaving residue name unchanged")
            continue
        old = residue.name
        residue.name = template
        renamed.append((label, old, template))
    for label, old, new in renamed:
        logging.info(f"Renamed glycan residue {label}: {old} -> {new}")
    logging.info(f"Renamed {len(renamed)} glycan residues to GLYCAM template names")


def validate_glycam_assignments(glycans_data: dict) -> None:
    assignments = glycans_data.get("glycam_assignments", {})
    unsupported = [
        (residue, data)
        for residue, data in sorted(assignments.items())
        if not data.get("template")
    ]

    if not unsupported:
        logging.info("All detected glycan residues have GLYCAM template assignments.")
        return

    logging.error("Unsupported glycan residues detected; cannot safely continue with GLYCAM preparation.")
    for residue, data in unsupported:
        logging.error(
            f"  {residue}: resname={data.get('resname')} "
            f"external_atoms={data.get('external_atoms')} reason={data.get('reason')}"
        )
    logging.error(
        "Add an explicit GLYCAM assignment rule for these residues/linkages, "
        "or supply a curated PDB with supported GLYCAM residue/template names."
    )
    sys.exit(1)


def build_nln_residue_template_map(topology: Topology, glycans_data: dict) -> dict:
    glyco_asn_labels = find_glycosylated_asn_labels_from_glycan_data(glycans_data)
    residue_templates = {}
    for residue in topology.residues():
        label_as_asn = f"ASN:{residue.chain.id}:{residue.id}"
        label_as_nln = f"NLN:{residue.chain.id}:{residue.id}"
        if label_as_asn in glyco_asn_labels or label_as_nln in glyco_asn_labels:
            ext = external_atoms_for_residue(topology, residue)
            if {"N", "C", "ND2"}.issubset(ext):
                template = "NLN"
            elif {"C", "ND2"}.issubset(ext) and "N" not in ext:
                template = "NNLN"
            elif {"N", "ND2"}.issubset(ext) and "C" not in ext:
                template = "CNLN"
            else:
                template = "NLN"
                logging.warning(
                    f"Could not infer NLN terminal variant for {residue.name}:{residue.chain.id}:{residue.id}; "
                    f"external_atoms={sorted(ext)}. Falling back to NLN."
                )
            residue_templates[residue] = template
            logging.info(
                f"Using explicit glycosylated ASN template "
                f"{residue.name}:{residue.chain.id}:{residue.id} -> {template}"
            )
    logging.info(f"Using {len(residue_templates)} explicit glycosylated ASN residue template assignments")
    return residue_templates


def expected_external_atoms_for_glycam_template(template: str) -> set[str] | None:
    expected = {
        "0YB": {"C1"}, "3YB": {"C1", "O3"}, "4YB": {"C1", "O4"},
        "6YB": {"C1", "O6"}, "UYB": {"C1", "O4", "O6"},
        "VYB": {"C1", "O3", "O6"}, "WYB": {"C1", "O3", "O4"}, "QYB": {"C1", "O3", "O4", "O6"},
        "0MA": {"C1"}, "3MA": {"C1", "O3"}, "4MA": {"C1", "O4"}, "6MA": {"C1", "O6"},
        "VMA": {"C1", "O3", "O6"}, "WMA": {"C1", "O3", "O4"}, "UMA": {"C1", "O4", "O6"}, "QMA": {"C1", "O3", "O4", "O6"},
        "0MB": {"C1"}, "3MB": {"C1", "O3"}, "4MB": {"C1", "O4"}, "6MB": {"C1", "O6"},
        "VMB": {"C1", "O3", "O6"}, "WMB": {"C1", "O3", "O4"}, "UMB": {"C1", "O4", "O6"}, "QMB": {"C1", "O3", "O4", "O6"},
        "0FA": {"C1"},
    }
    return expected.get(template)


def external_atoms_for_residue(topology: Topology, residue) -> set[str]:
    ext = set()
    for a1, a2 in topology.bonds():
        if a1.residue == residue and a2.residue != residue:
            ext.add(a1.name)
        elif a2.residue == residue and a1.residue != residue:
            ext.add(a2.name)
    return ext


def log_glycam_sanity_checks(modeller: Modeller, glycans_data: dict) -> None:
    assignments = glycans_data.get("glycam_assignments", {})
    assignment_by_chain_resid = {}
    for old_label, data in assignments.items():
        _, chain, resid = _glyco_parse_residue_label(old_label)
        assignment_by_chain_resid[(chain, resid)] = data

    passed = 0
    warnings = 0
    logging.info("GLYCAM topology sanity checks:")
    for residue in modeller.topology.residues():
        key = (residue.chain.id, residue.id)
        if key not in assignment_by_chain_resid:
            continue
        template = assignment_by_chain_resid[key].get("template")
        expected_atoms = assignment_by_chain_resid[key].get("expected_external_atoms") or []
        expected = set(expected_atoms) if expected_atoms else expected_external_atoms_for_glycam_template(template)
        found = external_atoms_for_residue(modeller.topology, residue)
        label = f"{residue.name}:{residue.chain.id}:{residue.id}"
        if expected is None:
            logging.warning(f"  {label}: no sanity rule for template {template}")
            warnings += 1
        elif expected == found:
            logging.info(f"  [OK] {label}: expected external={sorted(expected)} found={sorted(found)}")
            passed += 1
        else:
            logging.error(f"  [FAIL] {label}: expected external={sorted(expected)} found={sorted(found)}")
            warnings += 1
    logging.info(f"GLYCAM sanity checks passed for {passed} glycan residues")
    if warnings:
        logging.warning(f"{warnings} glycan sanity checks need inspection")


def dump_glyco_debug_files(modeller: Modeller, residue_templates: dict, outdir: Path, target_index: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    debug_pdb = outdir / "debug_before_hydrogens.pdb"
    debug_residue = outdir / f"debug_residue_{target_index}.txt"
    debug_templates = outdir / "debug_template_residues.txt"

    with open(debug_pdb, "w") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    with open(debug_residue, "w") as out:
        wrote = False
        for residue in modeller.topology.residues():
            if residue.index != target_index:
                continue
            wrote = True
            out.write(f"Residue index: {residue.index}\n")
            out.write(f"Residue: {residue.name}:{residue.chain.id}:{residue.id}\n\n")
            out.write("Atoms:\n")
            for atom in residue.atoms():
                out.write(f"  {atom.index:5d} {atom.name}\n")
            out.write("\nBonds involving this residue:\n")
            for a1, a2 in modeller.topology.bonds():
                if a1.residue == residue or a2.residue == residue:
                    out.write(f"  {a1.residue.name}:{a1.residue.chain.id}:{a1.residue.id}:{a1.name} -- {a2.residue.name}:{a2.residue.chain.id}:{a2.residue.id}:{a2.name}\n")
            break
        if not wrote:
            out.write(f"No residue with topology index {target_index} found.\n")

    with open(debug_templates, "w") as out:
        for residue, template in residue_templates.items():
            out.write("=" * 80 + "\n")
            out.write(f"Residue index: {residue.index}\nResidue: {residue.name}:{residue.chain.id}:{residue.id}\nTemplate: {template}\n\n")
            out.write("Atoms:\n")
            for atom in residue.atoms():
                out.write(f"  {atom.index:5d} {atom.name}\n")
            out.write("\nBonds involving this residue:\n")
            for a1, a2 in modeller.topology.bonds():
                if a1.residue == residue or a2.residue == residue:
                    out.write(f"  {a1.residue.name}:{a1.residue.chain.id}:{a1.residue.id}:{a1.name} -- {a2.residue.name}:{a2.residue.chain.id}:{a2.residue.id}:{a2.name}\n")
            out.write("\n")

    logging.info(f"Wrote glycoprotein debug PDB: {debug_pdb}")
    logging.info(f"Wrote glycoprotein debug residue dump: {debug_residue}")
    logging.info(f"Wrote glycoprotein debug template dump: {debug_templates}")


def prepare_glycoprotein_modeller(
    args: argparse.Namespace,
    forcefield: ForceField,
    glyco_outdir: Path,
) -> tuple[Modeller, dict]:
    glycans_data = detect_glycans_for_prepare(
        args.receptor,
        glyco_outdir,
        forcefield,
        use_link_records=not args.glyco_no_link_records,
        use_distance_detection=not args.glyco_no_distance_detection,
    )
    pdbfixer_input = write_pdb_with_supported_glycan_links_only(
        args.receptor,
        glyco_outdir / "pdbfixer_supported_links.pdb",
    )

    fixer = prepare_protein(
        pdbfixer_input,
        ncaa=args.ncaa,
        ncaa_resnames=args.ncaa_resnames,
    )
    modeller = Modeller(fixer.topology, fixer.positions)

    validate_glycam_assignments(glycans_data)
    rename_glycosylated_asn_to_nln(modeller, glycans_data)
    rename_glycan_atoms_to_glycam(modeller)
    rename_glycan_residues_to_glycam_templates(modeller, glycans_data)

    if args.ncaa:
        patch_ncaa_internal_bonds_from_ffxml(modeller.topology, args.ncaa_resnames, args.ncaa_ffxml)
        patch_ncaa_peptide_bonds(modeller.topology, args.ncaa_resnames)

    residue_templates = build_nln_residue_template_map(modeller.topology, glycans_data)
    log_glycam_sanity_checks(modeller, glycans_data)

    if args.glyco_debug:
        dump_glyco_debug_files(modeller, residue_templates, glyco_outdir, args.glyco_debug_residue_index)

    variants = None
    if args.metal_aware_protonation:
        variants = build_metal_aware_protonation_variants(
            modeller.topology,
            modeller.positions,
            args.metal_coordination_cutoff,
        )

    logging.info("Adding hydrogens using Modeller/ForceField templates in GLYCAM mode...")
    modeller.addHydrogens(forcefield, pH=args.ph, variants=variants, residueTemplates=residue_templates)
    logging.info(f"Atoms after GLYCAM hydrogen addition: {modeller.topology.getNumAtoms()}")

    return modeller, glycans_data

