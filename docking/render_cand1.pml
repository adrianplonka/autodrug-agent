# Render cand1 (best EGFR docking pose) in the EGFR ATP pocket.
# Run headless:  pymol.exe -cq render_cand1.pml
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

load 1M17_rec.pdb, rec
load cand1_pose1.sdf, cand1
hide everything

# receptor cartoon backdrop
show cartoon, rec and polymer
color grey70, rec and polymer

# binding-pocket residues (within 4.5 A of ligand)
select pocket, byres (rec and polymer within 4.5 of cand1)
show sticks, pocket and sidechain
set stick_radius, 0.13, pocket
color palecyan, pocket and elem C
util.cnc pocket

# ligand: the focus
show sticks, cand1
set stick_radius, 0.22, cand1
color yellow, cand1 and elem C
util.cnc cand1

# polar contacts (H-bonds) ligand <-> pocket
distance hbonds, cand1, (rec and polymer), 3.6, mode=2
color orange, hbonds

# label residues making those contacts
select hbres, byres ((rec and polymer) within 3.6 of cand1) and name CA
label hbres, "%s%s" % (resn, resi)
set label_position, (0, 0, 2)

# view & render: overview
orient cand1
zoom cand1, 5.0
turn y, 15
ray 1600, 1200
png cand1_EGFR_pocket.png, dpi=200

# close-up
zoom cand1, 2.5
ray 1600, 1200
png cand1_EGFR_closeup.png, dpi=200
