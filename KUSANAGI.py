import os, re, sys, json, tempfile, webbrowser, csv
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple

import easygui
import numpy as np
import cclib
from cclib.parser.utils import PeriodicTable

import math

def _open_canvas_window(title="KUSANAGI 3D (fast)", size=(900, 650)):
    if tk._default_root is None:
        root = tk.Tk()
        root.withdraw()
    win = tk.Toplevel()
    win.title(title)
    w, h = size
    canvas = tk.Canvas(win, width=w, height=h, bg="white", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    return win, canvas


def _bring_window_front(win):
    try:
        win.update_idletasks()
        win.lift()
        win.attributes("-topmost", True)
        win.after(150, lambda: win.attributes("-topmost", False))
        win.update()
    except Exception:
        pass


def _set_axes_equal(ax):
    import numpy as np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5*max([x_range, y_range, z_range])
    x_mid = 0.5*sum(x_limits); y_mid = 0.5*sum(y_limits); z_mid = 0.5*sum(z_limits)
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])



def _style_axes_3d(ax):
    ax.grid(False)
    ax.set_axis_off()



SMD_MENU_SOLVENTS = [
    "Acetonitrile",
    "Dichloromethane",
    "Methanol",
    "Water",
    "Toluene",
    "THF",
    "Chloroform",
]


SMD_KNOWN = set([
    "Water", "Acetonitrile", "Methanol", "Ethanol", "IsoQuinoline",
    "Quinoline", "Chloroform", "DiethylEther", "Dichloromethane",
    "DiChloroEthane", "CarbonTetraChloride", "Benzene", "Toluene",
    "ChloroBenzene", "NitroMethane", "Heptane", "CycloHexane",
    "Aniline", "Acetone", "TetraHydroFuran", "DiMethylSulfoxide",
    "Argon", "Krypton", "Xenon", "n-Octanol", "1,1,1-TriChloroEthane",
    "1,1,2-TriChloroEthane", "1,2,4-TriMethylBenzene", "1,2-DiBromoEthane",
    "1,2-EthaneDiol", "1,4-Dioxane", "1-Bromo-2-MethylPropane",
    "1-BromoOctane", "1-BromoPentane", "1-BromoPropane", "1-Butanol",
    "1-ChloroHexane", "1-ChloroPentane", "1-ChloroPropane", "1-Decanol",
    "1-FluoroOctane", "1-Heptanol", "1-Hexanol", "1-Hexene", "1-Hexyne",
    "1-IodoButane", "1-IodoHexaDecane", "1-IodoPentane", "1-IodoPropane",
    "1-NitroPropane", "1-Nonanol", "1-Pentanol", "1-Pentene", "1-Propanol",
    "2,2,2-TriFluoroEthanol", "2,2,4-TriMethylPentane", "2,4-DiMethylPentane",
    "2,4-DiMethylPyridine", "2,6-DiMethylPyridine", "2-BromoPropane",
    "2-Butanol", "2-ChloroButane", "2-Heptanone", "2-Hexanone",
    "2-MethoxyEthanol", "2-Methyl-1-Propanol", "2-Methyl-2-Propanol",
    "2-MethylPentane", "2-MethylPyridine", "2-NitroPropane", "2-Octanone",
    "2-Pentanone", "2-Propanol", "2-Propen-1-ol", "3-MethylPyridine",
    "3-Pentanone", "4-Heptanone", "4-Methyl-2-Pentanone", "4-MethylPyridine",
    "5-Nonanone", "AceticAcid", "AcetoPhenone", "a-ChloroToluene",
    "Anisole", "Benzaldehyde", "BenzoNitrile", "BenzylAlcohol",
    "BromoBenzene", "BromoEthane", "Bromoform", "Butanal", "ButanoicAcid",
    "Butanone", "ButanoNitrile", "ButylAmine", "ButylEthanoate",
    "CarbonDiSulfide", "Cis-1,2-DiMethylCycloHexane", "Cis-Decalin",
    "CycloHexanone", "CycloPentane", "CycloPentanol", "CycloPentanone",
    "Decalin-mixture", "DiBromomEthane", "DiButylEther", "DiEthylAmine",
    "DiEthylSulfide", "DiIodoMethane", "DiIsoPropylEther",
    "DiMethylDiSulfide", "DiPhenylEther", "DiPropylAmine",
    "e-1,2-DiChloroEthene", "e-2-Pentene", "EthaneThiol", "EthylBenzene",
    "EthylEthanoate", "EthylMethanoate", "EthylPhenylEther",
    "FluoroBenzene", "Formamide", "FormicAcid", "HexanoicAcid",
    "IodoBenzene", "IodoEthane", "IodoMethane", "IsoPropylBenzene",
    "m-Cresol", "Mesitylene", "MethylBenzoate", "MethylButanoate",
    "MethylCycloHexane", "MethylEthanoate", "MethylMethanoate",
    "MethylPropanoate", "m-Xylene", "n-ButylBenzene", "n-Decane",
    "n-Dodecane", "n-Hexadecane", "n-Hexane", "NitroBenzene", "NitroEthane",
    "n-MethylAniline", "n-MethylFormamide-mixture", "n,n-DiMethylAcetamide",
    "n,n-DiMethylFormamide", "n-Nonane", "n-Octane", "n-Pentadecane",
    "n-Pentane", "n-Undecane", "o-ChloroToluene", "o-Cresol",
    "o-DiChloroBenzene", "o-NitroToluene", "o-Xylene", "Pentanal",
    "PentanoicAcid", "PentylAmine", "PentylEthanoate", "PerFluoroBenzene",
    "p-IsoPropylToluene", "Propanal", "PropanoicAcid", "PropanoNitrile",
    "PropylAmine", "PropylEthanoate", "p-Xylene", "Pyridine",
    "sec-ButylBenzene", "tert-ButylBenzene", "TetraChloroEthene",
    "TetraHydroThiophene-s,s-dioxide", "Tetralin", "Thiophene",
    "Thiophenol", "trans-Decalin", "TriButylPhosphate", "TriChloroEthene",
    "TriEthylAmine", "Xylene-mixture", "z-1,2-DiChloroEthene",
])


def _validate_solvent_exists(name: str) -> bool:
    if name in SMD_MENU_SOLVENTS:
        return True
    return any(name.lower() == s.lower() for s in SMD_KNOWN)


FUNCTIONALS_MENU = [
    "b3lyp",
    "cam-b3lyp",
    "m06-2x",
    "pbe0",
    "wb97xd",
]


BASIS_MENU = [
    "def2svp",
    "def2tzvpp",
    "cc-pvdz",
    "6-31g(d)",
    "6-31+g(d,p)",
    "6-311g(d,p)",
]


def _map_functional(fn: str) -> str:
    s = fn.strip().lower()
    if s in ("wb97xd", "wb97x-d", "wb97x d", "wb97x_d"):
        return "wb97xd"
    return s


def _extract_geom_charge_mult(path):
    data = cclib.io.ccread(path)
    if data is None or not hasattr(data, "atomcoords") or not hasattr(data, "atomnos"):
        raise RuntimeError("No geometry found in the file.")
    coords = data.atomcoords[-1]
    ptab = PeriodicTable()
    symbols = [ptab.element[Z] for Z in data.atomnos]
    charge = getattr(data, "charge", None)
    mult   = getattr(data, "mult",   None)

    if charge is None:
        try: charge = int(input("Total charge (e.g., 0): ").strip())
        except: charge = 0
    if mult is None:
        try: mult = int(input("Multiplicity (1 singlet, 2 doublet, ...): ").strip())
        except: mult = 1
    return symbols, coords.tolist(), int(charge), int(mult)


def _build_xyz_lines(symbols, coords):
    return "\n".join(f"{s:2s} {x: .6f} {y: .6f} {z: .6f}" for s,(x,y,z) in zip(symbols, coords))


def _choose(prompt, options, default_idx=0):
    print("\n" + prompt)
    for i, opt in enumerate(options, 1):
        print(f"[{i}] {opt}")
    raw = input(f"Select [default {default_idx+1}]: ").strip()
    if raw == "": return options[default_idx]
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options): return options[idx]
    except: pass
    return options[default_idx]

def _yesno(prompt, default="y"):
    raw = input(f"{prompt} [{'Y/n' if default.lower()=='y' else 'y/N'}]: ").strip().lower()
    if raw == "": return default.lower() == "y"
    return raw in ("y","yes")

def _solvent_keyword(name):
    mapping = {
        "Acetonitrile":"Acetonitrile","Dichloromethane":"Dichloromethane","Methanol":"Methanol",
        "Water":"Water","Toluene":"Toluene","THF":"THF","Chloroform":"Chloroform"
    }
    return mapping.get(name, name)

def _route_section(job, functional, basis, use_scrf, solvent, use_gd3bj=False, nstates=50):
    functional = _map_functional(functional)
    basis = basis.strip().lower()

    if job == "Single-point":
        jobkw = "SP"
    elif job == "EPR":
        jobkw = "Prop=EPR"
    elif job == "UV-Vis":
        jobkw = f"td(NStates={int(nstates)})"
    else:
        jobkw = job

    parts = ["#p", jobkw, functional, basis]

    if use_gd3bj:
        parts.append("empiricalDispersion=gd3bj")

    if use_scrf and solvent:
        if not _validate_solvent_exists(solvent):
            print(f"[Warning] '{solvent}' may not be a valid Gaussian SMD solvent name.")
        parts.append(f"SCRF=(Solvent={solvent})")

    parts += ["gfinput", "iop(6/7=3)"]
    return " ".join(parts)


def _build_gjf(route, title, charge, mult, symbols, coords, nproc=20, mem_mb=50000):
    xyz = _build_xyz_lines(symbols, coords)
    return f"""%nprocshared={nproc}
%mem={mem_mb}MB
{route}

{title}

{charge} {mult}
{xyz}

"""


def _ensure_gjf_ext(path: str) -> str:
    return path if path.lower().endswith(".gjf") else (path + ".gjf")


def preview_geometry_3d(symbols, coords):
    labels = [str(i+1) for i in range(len(coords))]
    values = [0.0]*len(coords)
    _show_canvas_molecule(symbols, coords, labels, values, title="Molecule Preview", units="", decimals=0)


def _setup_gaussian_input_from_file(source_path):
    symbols, coords, charge, mult = _extract_geom_charge_mult(source_path)

    job = _choose("Select job type:", ["Single-point", "EPR", "UV-Vis"], default_idx=0)
    functional = _choose("Select DFT functional:", FUNCTIONALS_MENU, default_idx=0)
    basis = _choose("Select basis set:", BASIS_MENU, default_idx=1)

    use_gd3bj = _yesno("Add Empiricaldispersion=GD3BJ?", default="n")

    use_scrf = _yesno("Use solvent model (SMD)?", default="y")
    solvent = None
    if use_scrf:
        solvent = _choose("Select solvent:", SMD_MENU_SOLVENTS, default_idx=0)

    nstates = 50
    if job == "UV-Vis":
        try:
            nstates = int(input("Number of excited states (NStates) [default 50]: ").strip() or "50")
        except:
            nstates = 50

    title = input("Comment/title (leave empty for default): ").strip()
    if not title:
        tail = f" in {solvent}" if (use_scrf and solvent) else ""
        title = f"{job} at {_map_functional(functional)} {basis}{tail}"

    route = _route_section(job, functional, basis, use_scrf, solvent, use_gd3bj=use_gd3bj, nstates=nstates)
    gjf_text = _build_gjf(route, title, charge, mult, symbols, coords, nproc=20, mem_mb=50000)

    user_tag = input("Name tag for file (default 'SQX'): ").strip() or "SQX"
    fn = _map_functional(functional)
    bs = basis.strip().lower()
    disp_tag = "gd3bj" if use_gd3bj else "nodisp"
    scrf_tag = (f"SMD-{solvent}" if (use_scrf and solvent) else "")
    calc_tag = job
    default_name = f"{calc_tag}_{user_tag}_{fn}-{bs}_{disp_tag}_{scrf_tag}.gjf"

    outpath = easygui.filesavebox(title="Save Gaussian input as…", default=default_name, filetypes=["*.gjf","*.*"])
    if not outpath:
        print("Save cancelled.")
        return
    outpath = _ensure_gjf_ext(outpath)
    with open(outpath, "w", encoding="ascii", errors="ignore", newline="\n") as f:
        f.write(gjf_text)
    print(f"Saved: {outpath}")


def check_NImag():
    inpath = easygui.fileopenbox(title="Select Gaussian .log/.out with frequency job", default="*.log", filetypes=["*.log","*.out","*.*"])
    if not inpath:
        print("No file selected.")
        return

    nimag = None
    try:
        with open(inpath, "r", errors="ignore") as f:
            for line in f:
                if "NImag" in line:
                    m = re.search(r"NImag\s*=\s*(\d+)", line)
                    if m:
                        nimag = int(m.group(1))
                        break
    except Exception as e:
        print(f"[Error] Could not read file: {e}")
        return

    if nimag is None:
        print("No 'NImag' line found (wrong file or no frequency summary).")
        return

    print(f"NImag = {nimag}")

    try:
        symbols, coords = extract_geometry(inpath)
        preview_geometry_3d(symbols, coords)
    except Exception as e:
        print(f"[3D preview error] {e}")

    if nimag == 0:
        print("Verdict: Minimum (no imaginary frequencies).")
        if _yesno("Set up a Gaussian input for a subsequent calculation?", default="y"):
            try:
                _setup_gaussian_input_from_file(inpath)
            except Exception as e:
                print(f"[Setup error] {e}")
    elif nimag == 1:
        print("Verdict: Transition state (1 imaginary frequency).")
    else:
        print(f"Verdict: Higher-order saddle ({nimag} imaginary frequencies).")



SHELL_MAP = {"S":"s","P":"p","D":"d","F":"f","G":"g","SP":"sp","L":"sp"}

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    if np.isscalar(x): return [float(x)]
    try: return list(x)
    except Exception: return [float(x)]

def periodic_symbol(Z):
    try:
        from cclib.parser.utils import PeriodicTable
        return PeriodicTable().element[int(Z)]
    except Exception:
        return "X"

def fmt_D(x, prec=10):
    return f"{float(x):.{prec}E}".replace("E", "D")

def ensure_mo_rows_are_orbitals(mo):
    a = np.array(mo, dtype=float)
    return a if a.shape[0] >= a.shape[1] else a.T

def scalar_homo(homos, s, nmo):
    if isinstance(homos, (list, tuple, np.ndarray)):
        try:
            h = homos[s]
        except Exception:
            h = np.asarray(homos).max()
    else:
        h = homos
    h = int(np.asarray(h).astype(float).max())
    if h < 0: h = 0
    if h >= nmo: h = nmo - 1
    return h

def infer_occupations(data):
    if getattr(data, "mooccupations", None):
        occs = []
        for occ in data.mooccupations:
            arr = np.asarray(occ, dtype=float).ravel().tolist()
            occs.append(arr)
        return occs

    if not hasattr(data, "homos"):
        raise ValueError("Cannot infer occupations: 'homos' missing.")

    nspin = len(data.mocoeffs)
    occs = []
    for s in range(nspin):
        mo = np.asarray(data.mocoeffs[s])
        nmo = mo.shape[0] if mo.shape[0] == len(data.moenergies[s]) else mo.shape[1]
        h = scalar_homo(data.homos, s, nmo)
        occ = [1.0 if i <= h else 0.0 for i in range(nmo)]
        if nspin == 1:
            occ = [x * 2.0 for x in occ]
        occs.append(occ)
    return occs

def write_atoms_block(data):
    lines = ["[Atoms] Angs"]
    coords = data.atomcoords[-1]
    has_syms = hasattr(data, "atomsyms") and data.atomsyms is not None \
               and len(getattr(data, "atomsyms")) == len(data.atomnos)
    for i, (Z, xyz) in enumerate(zip(data.atomnos, coords), start=1):
        sym = (data.atomsyms[i-1] if has_syms else None) or periodic_symbol(Z)
        lines.append(f"{sym:>2s} {i:5d} {int(Z):3d} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}")
    return "\n".join(lines)

def write_5d7f_flag():
    return "[5D7F]"

def write_gto_block(data):
    gbasis = getattr(data, "gbasis", None)
    if gbasis is None:
        raise ValueError(
            "No basis set (gbasis) present. For Gaussian logs, run with Pop=Full GFInput; "
            "or load a .fchk created from the checkpoint."
        )

    out = ["[GTO]"]
    wrote_any_center = False

    for at_idx, shells in enumerate(gbasis, start=1):
        if not shells:
            continue

        out.append(f"{at_idx:4d} 0")
        wrote_any_center = True

        for shell in shells:
            stype = SHELL_MAP.get(str(shell[0]).upper(), str(shell[0]).lower())
            prims = shell[1]
            out.append(f"{stype:>2s} {len(prims):4d} 1.00")

            if stype == "sp":
                for exp, coeffs in prims:
                    cs_cp = to_list(coeffs)
                    if len(cs_cp) == 1: cs_cp = [cs_cp[0], 0.0]
                    elif len(cs_cp) > 2: cs_cp = cs_cp[:2]
                    out.append(f"  {fmt_D(exp):>16s}  {fmt_D(cs_cp[0]):>16s}  {fmt_D(cs_cp[1]):>16s}")
            else:
                for exp, coeffs in prims:
                    c_list = to_list(coeffs)
                    c = c_list[0] if c_list else 0.0
                    out.append(f"  {fmt_D(exp):>16s}  {fmt_D(c):>16s}")
        out.append("")

    if not wrote_any_center:
        raise ValueError("All centers had empty basis shells — cannot write [GTO].")
    return "\n".join(out).rstrip()

def write_mo_block(data):
    if not getattr(data, "mocoeffs", None):
        raise ValueError("No MO coefficients found.")
    if not getattr(data, "moenergies", None):
        raise ValueError("No MO energies found.")

    occs = infer_occupations(data)
    out = ["[MO]"]
    nspin = len(data.mocoeffs)
    spin_label = ["Alpha", "Beta"]

    for s in range(nspin):
        coeffs = ensure_mo_rows_are_orbitals(data.mocoeffs[s])
        energies = np.array(data.moenergies[s], dtype=float)
        if coeffs.shape[0] != energies.shape[0]:
            raise ValueError("Mismatch between #MOs in energies and coefficients.")
        occupations = occs[s]
        if len(occupations) != coeffs.shape[0]:
            occ_arr = np.zeros(coeffs.shape[0], dtype=float)
            occ_arr[:min(len(occupations), coeffs.shape[0])] = np.asarray(occupations, dtype=float)[:coeffs.shape[0]]
            occupations = occ_arr.tolist()

        for imo in range(coeffs.shape[0]):
            out.append(f" Ene= {energies[imo]:10.6f}")
            out.append(f" Spin= {spin_label[s]}")
            out.append(f" Occup= {float(occupations[imo]):10.6f}")
            for iao, c in enumerate(coeffs[imo, :], start=1):
                out.append(f"{iao:5d} {float(c):14.8f}")
            out.append("")
    return "\n".join(out).rstrip()

def build_molden_text(data):
    parts = [
        "[Molden Format]",
        write_atoms_block(data),
        "",
        write_5d7f_flag(),
        write_gto_block(data),
        "",
        write_mo_block(data),
    ]
    text = "\n".join(parts) + "\n"
    text = text.lstrip("\ufeff\r\n\t ")
    if not text.startswith("[Molden Format]"):
        text = "[Molden Format]\n" + text
    return text

def convert_gaussian_tomolden():
    inpath = easygui.fileopenbox(title="Select Gaussian .log/.out", default="*.log", filetypes=["*.log","*.out","*.*"])
    if not inpath:
        print("No file selected.")
        return
    try:
        data = cclib.io.ccread(inpath)
        if data is None:
            raise ValueError("Could not parse the selected file with cclib.")
    except Exception as e:
        print(f"[Parse error] {e}")
        return

    try:
        ptab = PeriodicTable()
        symbols = [ptab.element[Z] for Z in data.atomnos]
        coords = data.atomcoords[-1].tolist()

        win, _canvas, _upd = _show_canvas_molecule(
            symbols, coords,
            labels=[str(i+1) for i in range(len(coords))],
            values=[0.0]*len(coords),
            title="Molecule Preview",
            units="",
            decimals=0
        )
        try:
            win.update_idletasks()
            win.lift()
            win.attributes("-topmost", True)
            win.after(150, lambda: win.attributes("-topmost", False))
            win.update()
        except Exception:
            pass

        messagebox.showinfo(
            "3D Preview",
            "Die 3D-Ansicht ist geöffnet.\n"
            "Klicke auf OK, um den Export-Dialog zu öffnen."
        )
    except Exception as e:
        print(f"[3D preview failed] {e}")

    outpath = easygui.filesavebox(title="Save Molden file as…", default=os.path.splitext(os.path.basename(inpath))[0] + ".molden", filetypes=["*.molden","*.*"])
    if not outpath: return
    try:
        molden_txt = build_molden_text(data)#.lstrip("\ufeff\r\n\t ")
        with open(outpath, "w", encoding="ascii", errors="ignore", newline="\n") as f:
            f.write(molden_txt)
        with open(outpath, "rb") as f:
            head = f.read(20)
        if not head.startswith(b"[Molden Format]"):
            raise RuntimeError(f"Output does not start with [Molden Format]. First bytes: {head!r}")
        print(f"Saved Molden file: {outpath}")
    except Exception as e:
        import traceback
        print(f"[Write error] {e}\n{traceback.format_exc()}")
        return



def show_mulliken_charges():
    inpath = easygui.fileopenbox(
        title="Select Gaussian .log/.out (with Mulliken charges)",
        default="*.log",
        filetypes=["*.log","*.out","*.*"]
    )
    if not inpath:
        print("No file selected.")
        return
    try:
        parse_mulliken_charges(inpath)
    except Exception as e:
        print(f"[Invalid file] {e}")
        return


def parse_mulliken_charges(inpath):
    start_marker = " Mulliken charges"
    end_marker = " Sum of Mulliken charges"

    with open(inpath, 'r') as f:
        lines = f.readlines()

    start_index = end_index = None
    for i, line in enumerate(lines):
        if start_marker in line and start_index is None:
            start_index = i + 2
        elif end_marker in line and start_index is not None:
            end_index = i
            break

    if start_index is None or end_index is None:
        raise RuntimeError("File does not contain a Mulliken charge table.")

    table_lines = lines[start_index:end_index]
    table_data = [line.strip().split() for line in table_lines if line.strip()]

    print(f"\n{'Label':<6} {'Atom':<6} {'Charge':>10}")
    print("-" * 28)

    labels, atoms, charges = [], [], []
    for row in table_data:
        label = row[0]
        atom = row[1]
        charge = float(row[2])
        labels.append(label); atoms.append(atom); charges.append(charge)
        print(f"{label:<6} {atom:<6} {charge:10.4f}")

    symbols, coords = extract_geometry(inpath)
    win, canvas, _upd = _show_canvas_molecule(symbols, coords, labels=labels, values=charges, title="Mulliken charges", units="", decimals=3)
    _bring_window_front(win)



ELEMENT_COLORS = {
    "H": (1.0, 1.0, 1.0),   "He": (0.85, 1.0, 1.0),
    "Li": (0.8, 0.5, 1.0),  "Be": (0.76, 1.0, 0.0),
    "B": (1.0, 0.7, 0.7),   "C": (0.5, 0.5, 0.5),
    "N": (0.0, 0.0, 1.0),   "O": (1.0, 0.0, 0.0),
    "F": (0.0, 1.0, 0.0),   "Ne": (0.7, 0.89, 0.96),
    "Na": (0.67, 0.36, 0.95), "Mg": (0.54, 1.0, 0.0),
    "Al": (0.75, 0.65, 0.65), "Si": (0.94, 0.78, 0.63),
    "P": (1.0, 0.5, 0.0),   "S": (1.0, 1.0, 0.0),
    "Cl": (0.0, 1.0, 0.0),  "Ar": (0.5, 0.82, 0.89),
    "K": (0.56, 0.25, 0.83), "Ca": (0.24, 1.0, 0.0),
    "Sc": (0.9, 0.9, 0.9),  "Ti": (0.75, 0.76, 0.78),
    "V": (0.65, 0.65, 0.67), "Cr": (0.54, 0.6, 0.78),
    "Mn": (0.61, 0.48, 0.78), "Fe": (0.88, 0.4, 0.2),
    "Co": (0.94, 0.56, 0.63), "Ni": (0.31, 0.82, 0.31),
    "Cu": (0.78, 0.5, 0.2), "Zn": (0.49, 0.5, 0.69),
    "Ga": (0.76, 0.56, 0.56), "Ge": (0.4, 0.56, 0.56),
    "As": (0.74, 0.5, 0.89), "Se": (1.0, 0.63, 0.0),
    "Br": (0.6, 0.13, 0.0), "Kr": (0.36, 0.72, 0.82),
    "Rb": (0.43, 0.18, 0.69), "Sr": (0.0, 1.0, 0.0),
    "Y": (0.58, 1.0, 1.0), "Zr": (0.58, 0.88, 0.88),
    "Nb": (0.45, 0.76, 0.79), "Mo": (0.33, 0.71, 0.71),
    "Tc": (0.23, 0.62, 0.62), "Ru": (0.14, 0.56, 0.56),
    "Rh": (0.04, 0.49, 0.55), "Pd": (0.0, 0.41, 0.52),
    "Ag": (0.75, 0.75, 0.75), "Cd": (1.0, 0.85, 0.56),
    "In": (0.65, 0.46, 0.45), "Sn": (0.4, 0.5, 0.5),
    "Sb": (0.62, 0.39, 0.71), "Te": (0.83, 0.48, 0.0),
    "I": (0.6, 0.0, 1.0),  "Xe": (0.26, 0.62, 0.69),
}


def _color_hex_for(sym: str) -> str:
    rgb = ELEMENT_COLORS.get(sym, (0.7, 0.7, 0.7))
    try:
        r, g, b = rgb
        r = max(0, min(255, int(round(r*255))))
        g = max(0, min(255, int(round(g*255))))
        b = max(0, min(255, int(round(b*255))))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return str(rgb)


def show_normalized_spindensities():
    inpath = easygui.fileopenbox(title="Select Gaussian .log/.out (with Mulliken spin densities)", default="*.log", filetypes=["*.log","*.out","*.*"])
    if not inpath:
        print("No file selected.")
        return
    try:
        mulspi = "a"
        rows = normalize_mulliken_spin(inpath)
    except Exception as e:
        print(f"[Invalid file] {e}")
        return


def normalize_mulliken_spin(inpath):
    start_marker = " Mulliken charges and spin densities:"
    end_marker = " Sum of Mulliken charges"

    with open(inpath, 'r') as f:
        lines = f.readlines()

    start_index = end_index = None
    for i, line in enumerate(lines):
        if start_marker in line and start_index is None:
            start_index = i + 2
        elif end_marker in line and start_index is not None:
            end_index = i
            break

    if start_index is None or end_index is None:
        raise RuntimeError("File does not contain a Mulliken spin-density table.")

    table_lines = lines[start_index:end_index]
    table_data = [line.strip().split() for line in table_lines if line.strip()]

    norm_spin_dens = sum(abs(float(row[3])) for row in table_data)

    print(f"\n{'Label':<6} {'Atom':<6} {'Spin %':>7}")
    print("-" * 24)

    labels, atoms, percents = [], [], []
    for row in table_data:
        label = row[0]
        atom = row[1]
        spin_density = abs(float(row[3]))
        spin_percent = spin_density / norm_spin_dens * 100
        labels.append(label); atoms.append(atom); percents.append(spin_percent)
        print(f"{label:<6} {atom:<6} {spin_percent:7.1f}")

    symbols, coords = extract_geometry(inpath)
    win, canvas, _upd = _show_canvas_molecule(symbols, coords, labels=labels, values=percents, title="Normalized Mulliken spin densities", units="%", decimals=1)
    _bring_window_front(win)



def extract_geometry(log_path):
    data = cclib.io.ccread(log_path)
    if not hasattr(data, "atomcoords") or not hasattr(data, "atomnos"):
        raise RuntimeError("No coordinates found in this file.")
    coords = data.atomcoords[-1]
    ptab = PeriodicTable()
    symbols = [ptab.element[Z] for Z in data.atomnos]
    return symbols, coords.tolist()

def build_xyz(symbols, coords):
    return "\n".join(f"{s:2s} {x:.6f} {y:.6f} {z:.6f}" for s,(x,y,z) in zip(symbols, coords))

def gradient_color(p):
    t = max(0.0, min(1.0, p/100.0))
    r = int(255 * t); b = int(255 * (1.0-t)); g = 60
    return f"#{r:02x}{g:02x}{b:02x}"


def show_3d(symbols, coords, labels, values, units="%", decimals=1):
    _show_canvas_molecule(symbols, coords, labels, values, title="KUSANAGI 3D", units=units, decimals=decimals)



def draw_molecule_with_values(symbols, coords, labels, values, units="", decimals=2, title="KUSANAGI 3D"):
    return _show_canvas_molecule(symbols, coords, labels, values, units=units, decimals=decimals, title=title)



def _show_canvas_molecule(symbols, coords, labels, values, units="%", decimals=1, title="Molecule"):
    win, canvas = _open_canvas_window(title)

    bonds = perceive_bonds(symbols, coords, tol=0.45)
    state = {"elev":20, "azim":35, "zoom":1.0, "pan":[0,0], "drag":False, "mode":"rot", "mx":0, "my":0}

    current_vals = list(values)

    def redraw():
        w = int(canvas.winfo_width()); h = int(canvas.winfo_height())
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]; zs = [c[2] for c in coords]
        cx,cy,cz = (max(xs)+min(xs))/2.0,(max(ys)+min(ys))/2.0,(max(zs)+min(zs))/2.0
        centered=[(x-cx,y-cy,z-cz) for (x,y,z) in coords]

        a=math.radians(state["azim"]); e=math.radians(state["elev"])
        ca,sa,ce,se=math.cos(a),math.sin(a),math.cos(e),math.sin(e)
        Rx=((1,0,0),(0,ce,-se),(0,se,ce)); Rz=((ca,-sa,0),(sa,ca,0),(0,0,1))
        R=[[sum(Rz[i][k]*Rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        rot=[(R[0][0]*x+R[0][1]*y+R[0][2]*z,
              R[1][0]*x+R[1][1]*y+R[1][2]*z,
              R[2][0]*x+R[2][1]*y+R[2][2]*z) for (x,y,z) in centered]

        maxspan=max(max(abs(u) for u,_v,_w in rot), max(abs(v) for _u,v,_w in rot),1.0)
        scale=min((w-80)/(2*maxspan),(h-80)/(2*maxspan))*state["zoom"]
        proj=[(w/2+state["pan"][0]+x*scale,h/2+state["pan"][1]-y*scale,z) for (x,y,z) in rot]

        canvas.delete("all")

        for i,j,_ in bonds:
            xi,yi,_=proj[i]; xj,yj,_=proj[j]
            canvas.create_line(xi,yi,xj,yj,fill="gray20",width=2)
        order=sorted(range(len(proj)),key=lambda k:proj[k][2])
        for i in order:
            x,y,_=proj[i]
            canvas.create_oval(x-8,y-8,x+8,y+8,fill=_color_hex_for(symbols[i]),outline="")

        nlab = min(len(proj), 150)
        for i in range(nlab):
            x,y,_=proj[i]
            if units:
                suff=f"{current_vals[i]:.{decimals}f}{units}"
            else:
                suff=f"{current_vals[i]:.{decimals}f}"
            canvas.create_text(x,y-12,text=f"{labels[i]}{symbols[i]} {suff}",font=("TkDefaultFont",9))

    def on_press(e): state["drag"]=True; state["mx"],state["my"]=e.x,e.y; state["mode"]="rot" if e.num==1 else "pan"
    def on_release(e): state["drag"]=False
    def on_motion(e):
        if not state["drag"]: return
        dx,dy=e.x-state["mx"],e.y-state["my"]; state["mx"],state["my"]=e.x,e.y
        if state["mode"]=="rot":
            state["azim"]+=dx*0.4; state["elev"]-=dy*0.4; state["elev"]=max(-89,min(89,state["elev"]))
        else:
            state["pan"][0]+=dx; state["pan"][1]+=dy
        redraw()
    def on_wheel(e):
        delta=e.delta if hasattr(e,"delta") else (120 if e.num==4 else -120)
        factor=1.1 if delta>0 else 0.9; state["zoom"]=max(0.2,min(5.0,state["zoom"]*factor)); redraw()

    canvas.bind("<Configure>",lambda e:redraw())
    canvas.bind("<ButtonPress-1>",on_press); canvas.bind("<ButtonRelease-1>",on_release); canvas.bind("<B1-Motion>",on_motion)
    canvas.bind("<ButtonPress-3>",on_press); canvas.bind("<ButtonRelease-3>",on_release); canvas.bind("<B3-Motion>",on_motion)
    canvas.bind("<MouseWheel>",on_wheel); canvas.bind("<Button-4>",on_wheel); canvas.bind("<Button-5>",on_wheel)

    redraw()

    def update_values(new_vals):
        nonlocal current_vals
        current_vals = list(new_vals)
        redraw()

    try:
        win.update_idletasks()
        win.lift()
        win.attributes("-topmost", True)
        win.after(150, lambda: win.attributes("-topmost", False))
        win.update()
    except Exception:
        pass

    return win, canvas, update_values



def show_fermi_hyperfine():
    inpath = easygui.fileopenbox(
        title="Select Gaussian .log/.out (with EPR isotropic Fermi contact couplings)",
        default="*.log", filetypes=["*.log","*.out","*.*"]
    )
    if not inpath:
        print("No file selected.")
        return

    try:
        labels, atoms, mhz = parse_fermi_mhz(inpath)
    except Exception as e:
        print(f"[Invalid file] {e}")
        return

    try:
        symbols, coords = extract_geometry(inpath)
    except Exception as e:
        print(f"[3D error] {e}")
        return

    print("\nEPR Isotropic Fermi Contact Couplings [MHz]")
    print("-" * 44)
    print(f"{'Idx':<4} {'Label':<6} {'Atom':<8} {'MHz':>10}")
    print("-" * 44)
    for i, (L, A, v) in enumerate(zip(labels, atoms, mhz), start=1):
        print(f"{i:<4} {L:<6} {A:<8} {v:10.2f}")
    print("-" * 44)

    win, canvas, update_values = draw_molecule_with_values(symbols, coords, labels, mhz, units="", decimals=2, title="Fermi contact couplings [MHz]")
    _bring_window_front(win)

    current = mhz[:]

    print("\nPlease select molecules for averaging, example:")
    print("  1,3; 2,4    -> averages {1,3} and {2,4} each seperatly (1-based Indices)")
    print("Empty input = done.\n")

    while True:
        inp = input("Groups (empty = done): ").strip()
        if not inp:
            break
        try:
            parts = [p.strip() for p in inp.split(";") if p.strip()]
            label_groups = []
            for grp in parts:
                idxs = [int(x.strip()) for x in grp.split(",") if x.strip()]
                labs = [labels[i-1] for i in idxs if 1 <= i <= len(labels)]
                if labs:
                    label_groups.append(",".join(labs))
            groups_raw = ";".join(label_groups)
            _, new_vals = apply_averaging(labels, current, groups_raw)
            current = new_vals
            update_values(current)
            print("Updated:")
            for i, (L, A, v) in enumerate(zip(labels, atoms, current), start=1):
                print(f"{i:<4} {L:<6} {A:<8} {v:10.2f}")
        except Exception as e:
            print(f"[Input Error] {e}")

    print("\nFinal (possibly averaged) Fermi contact couplings [MHz]:")
    print("-" * 44)
    print(f"{'Label':<6} {'Atom':<8} {'MHz':>10}")
    print("-" * 44)
    for L, A, v in zip(labels, atoms, current):
        print(f"{L:<6} {A:<8} {v:10.2f}")
    print("-" * 44)



def parse_fermi_mhz(inpath):
    try:
        symbols, _ = extract_geometry(inpath)
        natoms = len(symbols)
    except Exception:
        natoms = None

    start_marker = "Isotropic Fermi Contact Couplings"
    header_hit = False
    rows = []

    with open(inpath, "r", errors="ignore") as f:
        lines = f.readlines()

    start_i = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_i = i
            break
    if start_i is None:
        raise RuntimeError("Could not find 'Isotropic Fermi Contact Couplings' in this file.")

    i = start_i + 2

    while i < len(lines):
        s = lines[i].rstrip("\n")
        if not s.strip():
            break
        if set(s.strip()) == set("-"):
            break
        if "Anisotropic" in s or "g tensor" in s:
            break

        parts = s.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                lbl = parts[0]
                atom = parts[1]
                mhz = float(parts[3])
                rows.append((lbl, atom, mhz))
            except Exception:
                pass
        i += 1
        if natoms is not None and len(rows) >= natoms:
            break

    if not rows:
        raise RuntimeError("Found the EPR section, but couldn't parse any rows.")

    labels = [r[0] for r in rows]
    atoms  = [r[1] for r in rows]
    mhz    = [r[2] for r in rows]
    return labels, atoms, mhz


def apply_averaging(labels, values, groups_raw):
    idx_by_label = {lab: i for i, lab in enumerate(labels)}
    vals = values[:]

    for grp in groups_raw.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        labs = [x.strip() for x in grp.split(",") if x.strip()]
        idxs = [idx_by_label[x] for x in labs if x in idx_by_label]
        if idxs:
            avg = float(sum(abs(vals[i]) for i in idxs)) / float(len(idxs))
            for i in idxs:
                vals[i] = avg

    return labels, vals





def _hide_builtin_picking_text(plotter):
    try:
        renderers = [plotter.renderer]
        if hasattr(plotter, "renderers"):
            renderers = list(plotter.renderers) or renderers
        from vtkmodules.vtkRenderingCore import vtkTextActor

        for ren in renderers:
            vtkren = getattr(ren, "_renderer", None) or getattr(ren, "renderer", None)
            if vtkren is None:
                continue
            actors2d = vtkren.GetActors2D()
            actors2d.InitTraversal()
            for _ in range(actors2d.GetNumberOfItems()):
                a = actors2d.GetNextActor2D()
                if isinstance(a, vtkTextActor):
                    txt = (a.GetInput() or "").lower()
                    if "pick" in txt or "press p" in txt or "left click" in txt:
                        a.SetVisibility(False)
        plotter.render()
    except Exception:
        pass


def interactive_group_average_values(symbols, coords, labels, values, units="", decimals=2, window_size=(1100, 800)):
    coords = np.asarray(coords, float)
    n = len(coords)
    orig_vals = values[:]
    vals = values[:]
    current_sel = set()
    groups = []
    history = []

    def fmt_val(v):
        return f"{v:.{decimals}f}{units}" if units else f"{v:.{decimals}f}"

    def label_text(i):
        return f"{labels[i]}{symbols[i]} {fmt_val(vals[i])}"

    pl = pv.Plotter(window_size=window_size)

    inst = "Click: select • A: average • U: undo • C: clear • R: reset • Q: finish"
    pl.add_text(inst, font_size=12, position=(10, 50), color='black')
    sel_text_actor = pl.add_text("Selected: []", font_size=12, position=(10, 10), color='black')

    sphere_radius = 0.30
    atom_meshes = []
    mesh2idx = {}
    atom_actors = []
    for i, ((x, y, z), sym) in enumerate(zip(coords, symbols)):
        color = ELEMENT_COLORS.get(sym, (0.7, 0.7, 0.7))
        mesh = pv.Sphere(radius=sphere_radius, center=(x, y, z))
        atom_meshes.append(mesh)
        mesh2idx[id(mesh)] = i
        actor = pl.add_mesh(mesh, color=color, pickable=True)
        atom_actors.append(actor)

    for i, j, order in perceive_bonds(symbols, coords, tol=0.45):
        if order == 1.5:
            lw, col = 5, "dimgray"
        elif order >= 2.5:
            lw, col = 8, "black"
        elif order >= 1.5:
            lw, col = 6, "black"
        else:
            lw, col = 3, "gray"
        pl.add_mesh(pv.Line(tuple(coords[i]), tuple(coords[j])), color=col, line_width=lw, pickable=False)

    label_actors = [None]*n

    def refresh_selection_text():
        if current_sel:
            ordered = sorted(int(i)+1 for i in current_sel) 
            pl.remove_actor(sel_text_actor, reset_camera=False)
            return pl.add_text(f"Selected: {ordered}", font_size=12, position='lower_left', color='black')
        else:
            pl.remove_actor(sel_text_actor, reset_camera=False)
            return pl.add_text("Selected: []", font_size=12, position='lower_left', color='black')

    def refresh_labels():
        nonlocal label_actors
        for a in label_actors:
            if a is not None:
                try: pl.remove_actor(a)
                except Exception: pass

        centroid = coords.mean(axis=0)
        bbox = coords.max(axis=0) - coords.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        offset = 0.025*diag if diag > 1e-6 else 0.25
        font_sz = 14 if diag < 15 else 16

        new_acts = []
        for i, (x,y,z) in enumerate(coords):
            v = np.array([x,y,z]) - centroid
            nrm = np.linalg.norm(v)
            nvec = v/nrm if nrm > 1e-6 else np.array([0,0,1.0])
            lx, ly, lz = (np.array([x,y,z]) + nvec*offset).tolist()
            txt = label_text(i)
            a = pl.add_point_labels([(lx,ly,lz)], [txt],
                                    font_size=font_sz,
                                    text_color='black',
                                    shape=None,
                                    always_visible=True,
                                    show_points=False,
                                    bold=(i in current_sel))
            new_acts.append(a)
        label_actors = new_acts
        pl.render()

    def refresh_atom_colors():
        for i, actor in enumerate(atom_actors):
            if i in current_sel:
                actor.prop.color = (1.0, 0.85, 0.2)
            else:
                actor.prop.color = ELEMENT_COLORS.get(symbols[i], (0.7,0.7,0.7))
        pl.render()

    def commit_current_group():
        nonlocal history, groups, current_sel, vals
        if not current_sel:
            return
        idxs = sorted(current_sel)
        prev = [vals[i] for i in idxs]
        mean_val = float(np.mean(np.abs([vals[i] for i in idxs])))
        for i in idxs:
            vals[i] = mean_val
        history.append((idxs, prev))
        groups.append(idxs)
        current_sel = set()
        refresh_atom_colors()
        refresh_labels()

    def undo_last_group():
        nonlocal history, groups, vals
        if not history:
            return
        idxs, prev = history.pop()
        for i, v in zip(idxs, prev):
            vals[i] = v
        if groups:
            groups.pop()
        refresh_atom_colors()
        refresh_labels()

    def clear_selection():
        nonlocal current_sel
        current_sel = set()
        refresh_atom_colors()
        refresh_labels()

    def reset_all():
        nonlocal vals, current_sel, history, groups
        vals = orig_vals[:]
        current_sel = set()
        history = []
        groups = []
        refresh_atom_colors()
        refresh_labels()

    def on_pick_mesh(mesh):
        if mesh is None:
            return
        i = mesh2idx.get(id(mesh))
        if i is None:
            return
        if i in current_sel:
            current_sel.remove(i)
        else:
            current_sel.add(i)
        refresh_atom_colors()
        refresh_labels()
        nonlocal sel_text_actor
        sel_text_actor = refresh_selection_text()

    pl.add_key_event("a", commit_current_group)
    pl.add_key_event("u", undo_last_group)
    pl.add_key_event("c", clear_selection)
    pl.add_key_event("r", reset_all)
    pl.add_key_event("q", lambda: pl.close())

    try:
        pl.enable_mesh_picking(callback=on_pick_mesh, left_clicking=True, show=False)
    except TypeError:
        pl.enable_mesh_picking(callback=on_pick_mesh, left_clicking=True, show_message=False)

    _hide_builtin_picking_text(pl)

    for actor in list(pl.renderer.actors.values()):
        try:
            if hasattr(actor, "GetInput"):
                txt = actor.GetInput() or ""
                if "press P to pick" in txt or "Left click" in txt:
                    actor.SetVisibility(False)
        except Exception:
            pass

    pl.reset_camera(); pl.camera.zoom(1.6)
    refresh_labels()
    sel_text_actor = refresh_selection_text()
    pl.show()

    return vals, groups


_BOND_REF = {
    ("C","C"):  {"1":1.54, "2":1.34, "3":1.20},
    ("C","N"):  {"1":1.47, "2":1.28, "3":1.16},
    ("C","O"):  {"1":1.43, "2":1.21, "3":1.13},
    ("N","O"):  {"1":1.40, "2":1.21},
    ("N","N"):  {"1":1.45, "2":1.25, "3":1.10},
    ("C","S"):  {"1":1.82, "2":1.56},
    ("C","F"):  {"1":1.35},
    ("C","Cl"): {"1":1.77},
    ("C","Br"): {"1":1.94},
    ("C","I"):  {"1":2.14},
}

COVALENT_RADII = {
    "H":0.31,"B":0.85,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"Si":1.11,"P":1.07,"S":1.05,
    "Cl":1.02,"Br":1.20,"I":1.39,
    "Li":1.28,"Na":1.66,"K":2.03,"Ca":1.74,
}

TYPICAL_MAX_VALENCE = {
    "H":1,"C":4,"N":3,"O":2,"F":1,"Cl":1,"Br":1,"I":1,"B":3,"P":3,"S":2,"Si":4,
}

def _cov(symbol): return COVALENT_RADII.get(symbol, 0.77)
def _maxval(symbol): return TYPICAL_MAX_VALENCE.get(symbol, 4)
def _pairkey(a,b): return tuple(sorted((a,b), key=str))

def _order_from_distance(sym_i, sym_j, d, ri, rj):
    key = _pairkey(sym_i, sym_j)
    ref = _BOND_REF.get(key)
    if ref:
        cuts = []
        if "3" in ref and "2" in ref:
            cuts.append(((ref["3"] + ref["2"]) * 0.5, 2.5))
        if "2" in ref and "1" in ref:
            cuts.append(((ref["2"] + ref["1"]) * 0.5, 1.5))
        cuts.sort(key=lambda x: x[0])
        if cuts:
            if d <= cuts[0][0]:
                return 3.0 if cuts[0][1] > 2 else 2.0
            if len(cuts) > 1 and d <= cuts[1][0]:
                return 2.0
        return 1.0
    shortfall = (ri + rj) - d
    if shortfall > 0.32: return 3.0
    if shortfall > 0.18: return 2.0
    return 1.0

def _find_six_cycles_C_only(symbols, adj):
    n = len(symbols); cycles = set()
    def dfs(start, curr, prev):
        if len(curr) > 6: return
        u = curr[-1]
        for v in adj[u]:
            if v == prev or symbols[v] != "C": continue
            if v == start and len(curr) == 6:
                cycles.add(tuple(sorted(curr)))
            elif v not in curr and len(curr) < 6:
                dfs(start, curr+[v], u)
    for i in range(n):
        if symbols[i] == "C":
            dfs(i, [i], -1)
    uniq, seen = [], set()
    for cyc in cycles:
        if cyc not in seen:
            seen.add(cyc); uniq.append(list(sorted(cyc)))
    return uniq

def _planarity_ok(coords, cyc, tol_deg=12.0):
    pts = np.asarray([coords[k] for k in cyc], float)
    normals = []
    p0 = pts[0]
    for a in range(1, len(pts)-1):
        v1 = pts[a] - p0; v2 = pts[a+1] - p0
        n = np.cross(v1, v2); nrm = np.linalg.norm(n)
        if nrm > 1e-6: normals.append(n/nrm)
    if len(normals) < 2: return True
    base = normals[0]
    for n in normals[1:]:
        ang = np.degrees(np.arccos(np.clip(np.dot(base, n), -1.0, 1.0)))
        if ang > tol_deg: return False
    return True

def perceive_bonds(symbols, coords, tol=0.45):
    xyz = np.asarray(coords, float); n = len(symbols)
    cand = []
    for i in range(n):
        ri = _cov(symbols[i])
        for j in range(i+1, n):
            rj = _cov(symbols[j])
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            if d < (ri + rj + tol):
                cand.append((i, j, d, ri, rj))
    if not cand: return []

    nbrs = [[] for _ in range(n)]
    for i, j, d, *_ in cand:
        nbrs[i].append((j, d)); nbrs[j].append((i, d))

    keep = set((min(i,j), max(i,j)) for i,j,_,_,_ in cand)
    for a in range(n):
        maxdeg = _maxval(symbols[a])
        lst = sorted(nbrs[a], key=lambda t: t[1])
        if len(lst) > maxdeg:
            for (b, _) in lst[maxdeg:]:
                keep.discard((min(a,b), max(a,b)))

    bonds = []; adj = [[] for _ in range(n)]
    for i, j, d, ri, rj in cand:
        e = (min(i,j), max(i,j))
        if e not in keep: continue
        order = _order_from_distance(symbols[i], symbols[j], d, ri, rj)
        bonds.append((i, j, order, d))
        adj[i].append(j); adj[j].append(i)

    cycles6 = _find_six_cycles_C_only(symbols, adj)
    if cycles6:
        edge2idx = { (min(i,j),max(i,j)) : k for k,(i,j,_,_) in enumerate(bonds) }
        for cyc in cycles6:
            edist = []; edges = []
            for a in range(6):
                u = cyc[a]; v = cyc[(a+1)%6]
                k = edge2idx.get((min(u,v), max(u,v)))
                if k is None: break
                edist.append(bonds[k][3]); edges.append(k)
            else:
                mean_d = float(np.mean(edist)); spread = float(np.std(edist))
                if 1.33 <= mean_d <= 1.45 and spread <= 0.06 and _planarity_ok(xyz, cyc):
                    for k in edges:
                        i, j, _, d = bonds[k]
                        bonds[k] = (i, j, 1.5, d)

    bonds.sort(key=lambda t: t[3])
    return [(i, j, order) for (i, j, order, _) in bonds]



MENU = """
=====================================================
                    KUSANAGI MENU
=====================================================
[1] Check for imaginary frequencies & generate input
[2] Convert Gaussian output to molden file
[3] Show Mulliken charges
[4] Show normalized spin densities
[5] Show Fermi hyperfine couplings
[Q] Quit
=====================================================
"""



def main():
    while True:
        print(MENU)
        choice = input("Select an option: ")
        if choice == "1":
            check_NImag()
        elif choice == "2":
            convert_gaussian_tomolden()
        elif choice == "3":
            show_mulliken_charges()
        elif choice == "4":
            show_normalized_spindensities()
        elif choice == "5":
            show_fermi_hyperfine()
        elif choice in {"q", "quit", "exit"}:
            print("Bye")
            break
        else:
            print("Unknown option.Please choose one of the valid options")

if __name__ == "__main__":
    main()
