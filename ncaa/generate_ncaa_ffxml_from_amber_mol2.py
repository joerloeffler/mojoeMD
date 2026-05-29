#!/usr/bin/env python3

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


DEFAULT_AMBER_TYPES = {
    "N": "protein-N",
    "H": "protein-H",
    "CA": "protein-CX",
    "C": "protein-C",
    "O": "protein-O",
    "CB": "protein-CT",
    "C01": "protein-CT",
}


DEFAULT_H_TYPES = {
    "H": "protein-H",
    "HB": "protein-HC",
    "H01": "protein-HC",
}

APPROXIMATE_MOL2_TYPE_MAP = {
    # These are compatibility fallbacks to ff14SB protein atom types, not a
    # substitute for a proper GAFF/OpenFF/Amber parameterization.
    "c": "protein-C",
    "c1": "protein-CT",
    "c2": "protein-CT",
    "c3": "protein-CT",
    "ca": "protein-CA",
    "cc": "protein-CC",
    "cd": "protein-CA",
    "ce": "protein-CA",
    "cf": "protein-CA",
    "cg": "protein-CA",
    "cp": "protein-CA",
    "cq": "protein-CA",
    "n": "protein-N",
    "n1": "protein-N2",
    "n2": "protein-N2",
    "n3": "protein-N3",
    "n4": "protein-N3",
    "na": "protein-NA",
    "nb": "protein-NB",
    "nc": "protein-NA",
    "nh": "protein-N2",
    "no": "protein-N2",
    "o": "protein-O",
    "oh": "protein-OH",
    "os": "protein-OH",
    "o2": "protein-O2",
    "s": "protein-S",
    "s2": "protein-S",
    "s4": "protein-S",
    "s6": "protein-S",
    "sh": "protein-SH",
    "ss": "protein-S",
    "h": "protein-H",
    "h1": "protein-H1",
    "h2": "protein-H1",
    "h3": "protein-H1",
    "h4": "protein-H4",
    "h5": "protein-H5",
    "ha": "protein-HA",
    "hc": "protein-HC",
    "hn": "protein-H",
    "ho": "protein-HO",
    "hs": "protein-HS",
    "hp": "protein-HP",
}

KNOWN_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "P", "S", "K", "V", "Y", "I", "W", "U",
    "CL", "BR", "NA", "MG", "AL", "SI", "CA", "MN", "FE", "CO", "NI", "CU", "ZN",
}

COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
}


STANDARD_BACKBONE_BONDS = [
    ("N", "CA", 1),
    ("CA", "C", 1),
    ("C", "O", 2),
]


def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def guess_pdb_element(atom_name, element_field=""):
    if element_field.strip():
        return element_field.strip().upper()

    name = atom_name.strip()
    if name and name[0].isdigit():
        name = name[1:]

    if len(name) >= 2 and name[:2].upper() in COVALENT_RADII:
        return name[:2].upper()

    return name[0].upper()


def parse_pdb_atoms(pdb_path: Path):
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atoms.append(
                {
                    "line": line.rstrip("\n"),
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resid": int(line[22:26]),
                    "icode": line[26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": guess_pdb_element(
                        line[12:16].strip(),
                        line[76:78] if len(line) >= 78 else "",
                    ),
                }
            )
    return atoms


def group_residues(atoms):
    residues = []
    current_key = None
    current = []

    for atom in atoms:
        key = (atom["chain"], atom["resid"], atom["icode"], atom["resname"])
        if current_key is None:
            current_key = key
        if key != current_key:
            residues.append({"key": current_key, "atoms": current})
            current_key = key
            current = []
        current.append(atom)

    if current:
        residues.append({"key": current_key, "atoms": current})

    return residues


def residue_label(residue):
    chain, resid, icode, resname = residue["key"]
    return f"{resname}{chain}{resid}{icode}"


def find_target_residue(residues, resname, resid=None, chain=None, icode=None):
    hits = []
    for i, residue in enumerate(residues):
        chain_i, resid_i, icode_i, resname_i = residue["key"]
        if resname_i != resname:
            continue
        if resid is not None and resid_i != resid:
            continue
        if chain is not None and chain_i != chain:
            continue
        if icode is not None and icode_i != icode:
            continue
        hits.append(i)

    if not hits:
        raise ValueError(f"No residue found matching resname={resname}, resid={resid}, chain={chain}, icode={icode}")
    if len(hits) > 1:
        labels = ", ".join(residue_label(residues[i]) for i in hits[:10])
        if len(hits) > 10:
            labels += ", ..."
        raise ValueError(
            "Multiple residues matched. Add --resid and/or --chain"
            f" to disambiguate. Matches: {labels}"
        )

    return hits[0]


def atom_dict(residue):
    return {atom["name"]: atom for atom in residue["atoms"]}


def pdb_distance(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        (a["z"] - b["z"]) ** 2
    )


def infer_distance_bonds(residue):
    atoms = residue["atoms"]
    bonds = []

    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            ea, eb = a["element"], b["element"]
            if ea == "H" and eb == "H":
                continue

            ra = COVALENT_RADII.get(ea)
            rb = COVALENT_RADII.get(eb)
            if ra is None or rb is None:
                continue

            d = pdb_distance(a, b)
            if d <= 1.25 * (ra + rb):
                order = 2 if {a["name"], b["name"]} == {"C", "O"} else 1
                bonds.append(
                    {
                        "residue": residue_label(residue),
                        "atom1": a["name"],
                        "atom2": b["name"],
                        "order": order,
                        "distance": round(d, 3),
                        "source": "distance",
                    }
                )

    return bonds


def add_backbone_bonds(residue):
    atoms = atom_dict(residue)
    out = []
    for a1, a2, order in STANDARD_BACKBONE_BONDS:
        if a1 in atoms and a2 in atoms:
            out.append(
                {
                    "residue": residue_label(residue),
                    "atom1": a1,
                    "atom2": a2,
                    "order": order,
                    "source": "backbone",
                }
            )
    return out


def add_peptide_bond(prev_residue, next_residue):
    prev_atoms = atom_dict(prev_residue)
    next_atoms = atom_dict(next_residue)

    if "C" not in prev_atoms or "N" not in next_atoms:
        return []

    return [
        {
            "residue": "PEPTIDE",
            "atom1_residue": residue_label(prev_residue),
            "atom1": "C",
            "atom2_residue": residue_label(next_residue),
            "atom2": "N",
            "order": 1,
            "source": "peptide",
        }
    ]


def unique_fragment_bonds(bonds):
    seen = set()
    out = []

    for bond in bonds:
        r1 = bond.get("atom1_residue", bond.get("residue"))
        r2 = bond.get("atom2_residue", bond.get("residue"))
        key = tuple(sorted([(r1, bond["atom1"]), (r2, bond["atom2"])]))
        if key in seen:
            continue
        seen.add(key)
        out.append(bond)

    return out


def write_fragment_pdb(fragment, out_pdb):
    serial = 1
    with open(out_pdb, "w") as out:
        for residue in fragment:
            for atom in residue["atoms"]:
                line = atom["line"]
                line = f"{line[:6]}{serial:5d}{line[11:]}"
                out.write(line + "\n")
                serial += 1
        out.write("TER\nEND\n")


def extract_charge_fragment(residues, target_idx, resname, resid, chain, outdir, context):
    start = max(0, target_idx - context)
    end = min(len(residues), target_idx + context + 1)
    fragment = residues[start:end]

    out_pdb = outdir / f"{resname}_fragment.pdb"
    out_json = outdir / f"{resname}_fragment_bonds.json"

    write_fragment_pdb(fragment, out_pdb)

    bonds = []
    for residue in fragment:
        bonds.extend(add_backbone_bonds(residue))
        bonds.extend(infer_distance_bonds(residue))

    for residue1, residue2 in zip(fragment[:-1], fragment[1:]):
        bonds.extend(add_peptide_bond(residue1, residue2))

    bonds = unique_fragment_bonds(bonds)

    report = {
        "target": {
            "resname": resname,
            "resid": resid,
            "chain": chain,
            "matched_residue": residue_label(residues[target_idx]),
        },
        "fragment_residues": [residue_label(residue) for residue in fragment],
        "bonds": bonds,
    }

    out_json.write_text(json.dumps(report, indent=2))
    return out_pdb, out_json, report


def parse_mol2(mol2_path: Path):
    atoms = []
    bonds = []

    section = None
    with open(mol2_path) as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("@<TRIPOS>ATOM"):
                section = "ATOM"
                continue
            if line.startswith("@<TRIPOS>BOND"):
                section = "BOND"
                continue
            if line.startswith("@<TRIPOS>"):
                section = None
                continue

            if section == "ATOM" and line.strip():
                parts = line.split()
                atoms.append(
                    {
                        "idx": int(parts[0]),
                        "name": parts[1],
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "z": float(parts[4]),
                        "mol2_type": parts[5],
                        "subst_id": int(parts[6]),
                        "subst_name": parts[7],
                        "charge": float(parts[8]),
                    }
                )

            elif section == "BOND" and line.strip():
                parts = line.split()
                bonds.append(
                    {
                        "idx": int(parts[0]),
                        "a1": int(parts[1]),
                        "a2": int(parts[2]),
                        "order": parts[3],
                    }
                )

    return atoms, bonds


def get_residue_atoms_from_mol2(atoms, resname):
    residue_atoms = [a for a in atoms if a["subst_name"] == resname]
    if not residue_atoms:
        raise ValueError(f"No MOL2 atoms found with substructure name {resname}")
    return residue_atoms


def infer_element(atom):
    raw_mol2_type = atom.get("mol2_type", "").split(".")[0]
    mol2_type = raw_mol2_type.upper()
    atom_name = "".join(ch for ch in atom.get("name", "") if ch.isalpha()).upper()

    if raw_mol2_type:
        lower_mol2_type = raw_mol2_type.lower()
        if lower_mol2_type in {"cl", "br"}:
            return lower_mol2_type.title()
        if raw_mol2_type[0].islower() and raw_mol2_type[0].upper() in KNOWN_ELEMENTS:
            return raw_mol2_type[0].upper()

    for token in (atom_name, mol2_type):
        if len(token) >= 2 and token[:2] in KNOWN_ELEMENTS and token[:2] not in {"CA"}:
            return token[:2].title()
        if token[:1] in KNOWN_ELEMENTS:
            return token[:1].title()

    raise ValueError(f"Could not infer element for MOL2 atom {atom.get('name')}")


def is_hydrogen(atom):
    return infer_element(atom).upper() == "H" or atom["name"].upper().startswith("H")


def distance(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def heavy_atoms_from_residue(residue):
    atoms = []
    for atom in residue["atoms"]:
        if atom["element"] == "H" or atom["name"].startswith("H"):
            continue
        atoms.append(
            {
                "name": atom["name"],
                "resname": atom["resname"],
                "chain": atom["chain"],
                "resid": atom["resid"],
                "x": atom["x"],
                "y": atom["y"],
                "z": atom["z"],
                "element": atom["element"],
            }
        )

    if not atoms:
        raise ValueError(f"No heavy atoms found for {residue_label(residue)}")

    return atoms


def map_heavy_atoms_by_coordinate(pdb_atoms, mol2_res_atoms, tolerance=0.15):
    mapping = {}

    heavy_mol2 = [a for a in mol2_res_atoms if not is_hydrogen(a)]
    used_mol2_indices = set()

    for pdb_atom in pdb_atoms:
        best = None
        best_d = 999.0

        for mol2_atom in heavy_mol2:
            if mol2_atom["idx"] in used_mol2_indices:
                continue
            d = distance(pdb_atom, mol2_atom)
            if d < best_d:
                best = mol2_atom
                best_d = d

        if best is None or best_d > tolerance:
            raise ValueError(
                f"Could not map PDB atom {pdb_atom['name']} by coordinates. "
                f"Best distance was {best_d:.3f} Å"
            )

        mapping[pdb_atom["name"]] = best
        used_mol2_indices.add(best["idx"])

    return mapping


def make_unique_name(name, used):
    base = name[:4]
    if base not in used:
        used.add(base)
        return base

    for i in range(1, 1000):
        suffix = str(i)
        candidate = f"{base[:4 - len(suffix)]}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate

    raise ValueError(f"Could not make unique atom name from {name}")


def build_template_atom_name_map(heavy_mapping, mol2_res_atoms, bonds):
    """Return {mol2_idx: template_atom_name} for all atoms in the MOL2 residue."""
    used_names = set()
    name_by_idx = {}

    for pdb_name, mol2_atom in heavy_mapping.items():
        name_by_idx[mol2_atom["idx"]] = make_unique_name(pdb_name, used_names)

    atom_by_idx = {a["idx"]: a for a in mol2_res_atoms}
    parent_by_h_idx = {}
    for b in bonds:
        a1 = atom_by_idx.get(b["a1"])
        a2 = atom_by_idx.get(b["a2"])
        if a1 is None or a2 is None:
            continue
        if is_hydrogen(a1) and not is_hydrogen(a2):
            parent_by_h_idx[a1["idx"]] = a2["idx"]
        elif is_hydrogen(a2) and not is_hydrogen(a1):
            parent_by_h_idx[a2["idx"]] = a1["idx"]

    for atom in mol2_res_atoms:
        if atom["idx"] in name_by_idx:
            continue

        if is_hydrogen(atom):
            parent_name = name_by_idx.get(parent_by_h_idx.get(atom["idx"]))
            if parent_name == "N" and "H" not in used_names:
                candidate = "H"
            elif parent_name:
                candidate = atom["name"]
            else:
                candidate = atom["name"]
        else:
            candidate = atom["name"]

        name_by_idx[atom["idx"]] = make_unique_name(candidate, used_names)

    return name_by_idx


def resolve_atom_type(atom, template_name, atom_type_map, type_source, mol2_type_prefix, allow_approximate_types):
    if template_name in atom_type_map:
        return atom_type_map[template_name]
    if atom["name"] in atom_type_map:
        return atom_type_map[atom["name"]]
    if atom["mol2_type"] in atom_type_map:
        return atom_type_map[atom["mol2_type"]]

    if type_source == "mol2":
        return f"{mol2_type_prefix}{atom['mol2_type']}"

    if template_name in DEFAULT_AMBER_TYPES:
        return DEFAULT_AMBER_TYPES[template_name]
    if template_name == "H":
        return DEFAULT_H_TYPES["H"]

    mol2_key = atom["mol2_type"].lower()
    element = infer_element(atom)
    if element.upper() == "H" and mol2_key in APPROXIMATE_MOL2_TYPE_MAP:
        return APPROXIMATE_MOL2_TYPE_MAP[mol2_key]
    if allow_approximate_types and mol2_key in APPROXIMATE_MOL2_TYPE_MAP:
        return APPROXIMATE_MOL2_TYPE_MAP[mol2_key]

    raise ValueError(
        f"No safe OpenMM atom type for {template_name} "
        f"(MOL2 atom {atom['name']}, type {atom['mol2_type']}, element {element}). "
        "AM1-BCC supplies charges only, not bonded/LJ parameters. Provide "
        "--atom-type-map for an ffxml that already defines suitable types, use "
        "--type-source mol2 with a matching GAFF/OpenFF-style ffxml, or rerun with "
        "--allow-approximate-types only if you accept ff14SB protein-type approximations."
    )


def load_atom_type_map(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def make_ffxml(
    resname,
    mol2_res_atoms,
    bonds,
    name_by_idx,
    atom_type_map,
    type_source="protein",
    mol2_type_prefix="gaff-",
    allow_approximate_types=False,
    target_charge=0.0,
    charge_correction="uniform",
):
    root = ET.Element("ForceField")
    residues = ET.SubElement(root, "Residues")
    residue = ET.SubElement(residues, "Residue", {"name": resname})

    template_atoms = []
    included_indices = {a["idx"] for a in mol2_res_atoms}

    for atom in sorted(mol2_res_atoms, key=lambda x: x["idx"]):
        name = name_by_idx[atom["idx"]]
        atype = resolve_atom_type(
            atom,
            name,
            atom_type_map,
            type_source,
            mol2_type_prefix,
            allow_approximate_types,
        )
        template_atoms.append((name, atype, atom["charge"], atom["idx"], infer_element(atom)))

    raw_total = sum(q for _, _, q, _, _ in template_atoms)
    if charge_correction != "none":
        delta = target_charge - raw_total
        if charge_correction == "uniform":
            correction_indices = list(range(len(template_atoms)))
        elif charge_correction == "heavy":
            correction_indices = [
                i for i, (_, _, _, _, element) in enumerate(template_atoms)
                if element.upper() != "H"
            ]
        else:
            raise ValueError(f"Unsupported charge correction mode: {charge_correction}")

        if not correction_indices:
            raise ValueError("No atoms available for charge correction.")

        correction = delta / len(correction_indices)
        corrected = []
        for i, (name, atype, charge, idx, element) in enumerate(template_atoms):
            if i in correction_indices:
                charge += correction
            corrected.append((name, atype, charge, idx, element))
        template_atoms = corrected

    for name, atype, charge, _, _ in template_atoms:
        ET.SubElement(
            residue,
            "Atom",
            {
                "name": name,
                "type": atype,
                "charge": f"{charge:.8f}",
            },
        )

    template_bonds = []
    external_bond_atoms = set()
    for bond in bonds:
        a1_in = bond["a1"] in included_indices
        a2_in = bond["a2"] in included_indices

        if a1_in and a2_in:
            template_bonds.append((name_by_idx[bond["a1"]], name_by_idx[bond["a2"]]))
        elif a1_in:
            external_bond_atoms.add(name_by_idx[bond["a1"]])
        elif a2_in:
            external_bond_atoms.add(name_by_idx[bond["a2"]])

    if not external_bond_atoms:
        for fallback_name in ("N", "C"):
            if fallback_name in set(name_by_idx.values()):
                external_bond_atoms.add(fallback_name)

    seen = set()
    for a1, a2 in template_bonds:
        key = tuple(sorted((a1, a2)))
        if key in seen:
            continue
        seen.add(key)
        ET.SubElement(residue, "Bond", {"atomName1": a1, "atomName2": a2})

    for atom_name in sorted(external_bond_atoms):
        ET.SubElement(residue, "ExternalBond", {"atomName": atom_name})

    return root, template_atoms, template_bonds, sorted(external_bond_atoms), raw_total


def make_hydrogen_xml(resname, mol2_res_atoms, bonds, name_by_idx):
    root = ET.Element("Residues")
    residue = ET.SubElement(root, "Residue", {"name": resname})
    atom_by_idx = {a["idx"]: a for a in mol2_res_atoms}

    hydrogen_parent_pairs = []
    for bond in bonds:
        a1 = atom_by_idx.get(bond["a1"])
        a2 = atom_by_idx.get(bond["a2"])
        if a1 is None or a2 is None:
            continue

        if is_hydrogen(a1) and not is_hydrogen(a2):
            hydrogen_parent_pairs.append((name_by_idx[a1["idx"]], name_by_idx[a2["idx"]]))
        elif is_hydrogen(a2) and not is_hydrogen(a1):
            hydrogen_parent_pairs.append((name_by_idx[a2["idx"]], name_by_idx[a1["idx"]]))

    for hname, parent in sorted(hydrogen_parent_pairs):
        ET.SubElement(residue, "H", {"name": hname, "parent": parent})

    return root


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenMM NCAA residue ffxml from AmberTools MOL2 charges."
    )
    parser.add_argument("--protein-pdb", required=True)
    parser.add_argument(
        "--mol2",
        default=None,
        help="AmberTools/antechamber MOL2 with AM1-BCC charges. If omitted with --extract-fragment, only the fragment is written.",
    )
    parser.add_argument("--resname", required=True)
    parser.add_argument("--resid", type=int, default=None)
    parser.add_argument("--chain", default=None)
    parser.add_argument("--icode", default=None)
    parser.add_argument("--outdir", default="ncaa")
    parser.add_argument(
        "--workdir",
        default="ncaa_work",
        help="Directory for extracted charge-fragment files.",
    )
    parser.add_argument(
        "--extract-fragment",
        action="store_true",
        help="Write the capped charge fragment PDB/JSON before ffxml generation.",
    )
    parser.add_argument(
        "--fragment-only",
        action="store_true",
        help="Only extract the capped charge fragment and stop.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=1,
        help="Number of neighboring residues to include on each side of the charge fragment.",
    )
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--no-neutralize", action="store_true")
    parser.add_argument(
        "--target-charge",
        type=float,
        default=0.0,
        help="Desired residue-template charge after correction. Use -1/-2 for charged NCAAs.",
    )
    parser.add_argument(
        "--charge-correction",
        choices=["uniform", "heavy", "none"],
        default="uniform",
        help="How to apply the residual AM1-BCC charge correction.",
    )
    parser.add_argument(
        "--atom-type-map",
        default=None,
        help=(
            "JSON mapping from template atom name, MOL2 atom name, or MOL2 type to an "
            "OpenMM atom type already defined by a loaded force-field XML."
        ),
    )
    parser.add_argument(
        "--type-source",
        choices=["protein", "mol2"],
        default="protein",
        help=(
            "protein: use ff14SB-compatible protein types where safe; "
            "mol2: emit prefixed MOL2/GAFF-like type names that must be defined elsewhere."
        ),
    )
    parser.add_argument(
        "--mol2-type-prefix",
        default="gaff-",
        help="Prefix used with --type-source mol2, e.g. gaff- + c3 -> gaff-c3.",
    )
    parser.add_argument(
        "--allow-approximate-types",
        action="store_true",
        help=(
            "Permit heuristic mapping of GAFF/MOL2 C/N/O/S/H types onto ff14SB protein "
            "types. This is not appropriate for phosphorus or novel chemistry."
        ),
    )
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)

    pdb_atoms_all = parse_pdb_atoms(Path(args.protein_pdb))
    residues = group_residues(pdb_atoms_all)
    target_idx = find_target_residue(
        residues,
        args.resname,
        args.resid,
        args.chain,
        args.icode,
    )
    target_residue = residues[target_idx]

    if args.extract_fragment or args.fragment_only or args.mol2 is None:
        workdir.mkdir(parents=True, exist_ok=True)
        fragment_pdb, fragment_json, fragment_report = extract_charge_fragment(
            residues,
            target_idx,
            args.resname,
            args.resid,
            args.chain,
            workdir,
            args.context,
        )
        print(f"Wrote: {fragment_pdb}")
        print(f"Wrote: {fragment_json}")
        print("Fragment:")
        for residue in fragment_report["fragment_residues"]:
            print(f"  {residue}")

    if args.fragment_only or args.mol2 is None:
        print()
        print("No --mol2 supplied, so stopping after fragment extraction.")
        print("Run AmberTools/antechamber on the fragment to produce an AM1-BCC MOL2, then rerun with --mol2.")
        return

    atoms, bonds = parse_mol2(Path(args.mol2))
    mol2_res_atoms = get_residue_atoms_from_mol2(atoms, args.resname)

    pdb_atoms = heavy_atoms_from_residue(target_residue)

    heavy_mapping = map_heavy_atoms_by_coordinate(
        pdb_atoms,
        mol2_res_atoms,
        tolerance=args.tolerance,
    )

    name_by_idx = build_template_atom_name_map(
        heavy_mapping,
        mol2_res_atoms,
        bonds,
    )
    atom_type_map = load_atom_type_map(args.atom_type_map)
    charge_correction = "none" if args.no_neutralize else args.charge_correction

    ffxml, template_atoms, template_bonds, external_bonds, raw_total_charge = make_ffxml(
        args.resname,
        mol2_res_atoms,
        bonds,
        name_by_idx,
        atom_type_map,
        type_source=args.type_source,
        mol2_type_prefix=args.mol2_type_prefix,
        allow_approximate_types=args.allow_approximate_types,
        target_charge=args.target_charge,
        charge_correction=charge_correction,
    )

    hxml = make_hydrogen_xml(args.resname, mol2_res_atoms, bonds, name_by_idx)
    template_hydrogens = [
        {"name": h.attrib["name"], "parent": h.attrib["parent"]}
        for h in hxml.findall(".//H")
    ]

    ffxml_path = outdir / f"{args.resname}.xml"
    hxml_path = outdir / f"{args.resname}_hydrogens.xml"
    report_path = outdir / f"{args.resname}_charge_report.json"

    for path in [ffxml_path, hxml_path, report_path]:
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists. Use --force to overwrite.")

    ffxml_path.write_text(prettify_xml(ffxml))
    hxml_path.write_text(prettify_xml(hxml))

    report = {
        "resname": args.resname,
        "heavy_mapping": {
            k: {
                "mol2_name": v["name"],
                "mol2_idx": v["idx"],
                "charge": v["charge"],
            }
            for k, v in heavy_mapping.items()
        },
        "template_hydrogens": template_hydrogens,
        "template_atom_name_by_mol2_idx": {
            str(k): v for k, v in sorted(name_by_idx.items())
        },
        "template_atoms": [
            {"name": n, "type": t, "charge": q, "mol2_idx": idx, "element": element}
            for n, t, q, idx, element in template_atoms
        ],
        "template_bonds": template_bonds,
        "external_bonds": external_bonds,
        "raw_total_charge": raw_total_charge,
        "target_charge": args.target_charge,
        "charge_correction": charge_correction,
        "total_template_charge": sum(q for _, _, q, _, _ in template_atoms),
    }

    report_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {ffxml_path}")
    print(f"Wrote {hxml_path}")
    print(f"Wrote {report_path}")
    print()
    print("Template atom charges:")
    for name, atype, charge, _, _ in template_atoms:
        print(f"  {name:5s} {atype:12s} {charge: .8f}")
    print(f"\nRaw total charge: {raw_total_charge: .8f}")
    print(f"Final total charge: {sum(q for _, _, q, _, _ in template_atoms): .8f}")


if __name__ == "__main__":
    main()
