# Render mol12 (best COX-2 docking pose) in the COX-2 binding pocket.
# Run headless:  pymol.exe -cq render_mol12.pml
reinitialize
bg_color white
set ray_opaque_background, 0
set ray_shadows, 0
set ray_trace_mode, 0
set ambient, 0.35
set cartoon_transparency, 0.6
set valence, 0
set dash_gap, 0.35
set dash_width, 3.0
set label_size, 15
set label_color, black

# --- load receptor + docked ligand ---------------------------------------
load 1CX2_rec.pdb, rec
load mol12_pose1.sdf, mol12

hide everything

# --- receptor cartoon (faded backdrop) -----------------------------------
show cartoon, rec and polymer
color grey70, rec and polymer

# --- heme for spatial context --------------------------------------------
show sticks, rec and resn HEM
color firebrick, rec and resn HEM
set stick_radius, 0.12, rec and resn HEM

# --- binding-pocket residues (within 4.5 A of ligand) --------------------
select pocket, byres (rec and polymer within 4.5 of mol12)
show sticks, pocket and sidechain
set stick_radius, 0.13, pocket
color palecyan, pocket and elem C
util.cnc pocket

# --- ligand: the focus ---------------------------------------------------
show sticks, mol12
set stick_radius, 0.22, mol12
color yellow, mol12 and elem C
util.cnc mol12

# --- polar contacts (H-bonds) ligand <-> pocket --------------------------
distance hbonds, mol12, (rec and polymer), 3.5, mode=2
color orange, hbonds

# --- label the residues making those contacts ----------------------------
select hbres, byres ((rec and polymer) within 3.5 of mol12) and name CA
label hbres, "%s%s" % (resn, resi)
set label_position, (0, 0, 2)

# --- view & render: overview ---------------------------------------------
orient mol12
zoom mol12, 5.0
turn y, 15
ray 1600, 1200
png mol12_COX2_pocket.png, dpi=200

# --- close-up on the ligand + contacts -----------------------------------
zoom mol12, 2.5
ray 1600, 1200
png mol12_COX2_closeup.png, dpi=200
