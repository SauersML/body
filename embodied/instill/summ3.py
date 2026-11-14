import json,glob,sys,numpy as np
pat=sys.argv[1]  # e.g. hypha-prod-b3-* 
R={}
for f in glob.glob(f'$BODY_WORK/runpod/{pat}/interp_*.json'):
    tag=f.split('interp_')[1][:-5]; m,a,s=tag.rsplit('_',2)
    t=json.load(open(f.replace('interp_','train_'))); i=json.load(open(f))
    R.setdefault((m,a),[]).append((t,i))
def L(i,key,which):  # which: 'inj','mid','max'
    d={k:v for k,v in i[key].items() if k not in('raw input',)}; ks=[k for k in d if k!='inject']
    return d['inject'] if which=='inj' else max(d[k] for k in ks)
for (m,a),runs in sorted(R.items()):
    f=lambda g: f"{np.mean(g):.2f}±{np.std(g):.2f}"
    cl=lambda k:[t['closed_loop'][k][0] for t,i in runs]
    ev=lambda k:[np.mean(list(t['eval'][k].values())) for t,i in runs if k in t['eval']]
    print(m,a,'n',len(runs),'| reach',f(cl('reach')),'find',f(cl('find')),'wall',f(cl('wall')),
      '| dim-actions',sum(sum(t['closed_loop'][k][1][13:16]) for t,i in runs for k in t['closed_loop']))
    print('   QA: pos MAE',f(ev('pos')),'x MAE',f(ev('x')),'self',f(ev('self')),'part/line',f(ev('part') or ev('line')))
    print('   boundary inj',f([i['boundary_contrast']['inject'] for t,i in runs]),'max layer',f([L(i,'boundary_contrast','max') for t,i in runs]),
      '| rsa inj',f([i['rsa_vs_skin']['inject'] for t,i in runs]),'| arc probe best',f([L(i,'arc_probe_r2','max') for t,i in runs]))
    print('   x probe inj',f([i['x_probe_r2']['inject'] for t,i in runs]),'best layer',f([L(i,'x_probe_r2','max') for t,i in runs]),
      '| x answer follows posture r',f([i['x_answer_follows_posture_r'] for t,i in runs]),'| self probe best',f([L(i,'self_probe_acc','max') for t,i in runs]))
    print('   pos answer range',f([np.ptp(i['pos_answer_curve']) for t,i in runs]),'staircase',f([i['staircase_index'] for t,i in runs]))
