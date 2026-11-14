import json,sys
L=sys.argv[1]; R=json.load(open(f'out/ana_{L}.json'))[L]
f=lambda x: f"{x[0]:.2f}[{x[1]:.2f},{x[2]:.2f}]" if isinstance(x,list) and len(x)==3 else str(x)
keys=['reg','surf','side','spot','spot_untouched','spot_newkinds','reg_untouched','surf_untouched','root_tip_rho','root_tip_rho_newkinds','rsa_wing_pos','rsa_wing_reg','rsa_wing_surf','rsa_wing_side','rsa_desc_pos','rsa_desc_reg','rsa_wing_rightwing_pos_reg','body_pback_root','body_pback_tip','body_pback_root-tip','body_ent_root','body_ent_tip','body_ent_root-tip','body_cosback_root','body_cosback_tip','body_cosback_root-tip','body_cosback_root-tip_rightwing','kd_drag-finger','kd_palm-finger','kd_cold-finger','kd_feather-finger','kd_warm-cold']
for k in keys:
    if k in R['A']: print(f"{k:26s}", "  ".join(f"{cn}:{f(R[cn].get(k))}" for cn in R))
for k in ['spot_vsA','spot_untouched_vsA','spot_newkinds_vsA','reg_vsA','surf_vsA','surf_untouched_vsA','body_cosback_root-tip_vsA','body_pback_root-tip_vsA','body_ent_root-tip_vsA','kd_drag-finger_vsA','kd_palm-finger_vsA','kd_cold-finger_vsA','kd_feather-finger_vsA']:
    if k in R['B']: print(f"{k:26s}", "  ".join(f"{cn}:{f(R[cn].get(k))}" for cn in R if cn!='A'))
