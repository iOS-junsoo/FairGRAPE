#!/usr/bin/env python3
"""Summarize completed smoke checkpoints; never executes a model."""
import argparse,csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def summarize(run):
    rows=[];epoch_rows=[];layer_rows=[]
    for arm in sorted(run.glob('alpha*')):
        if not arm.is_dir():continue
        for p in sorted(arm.glob('iteration_*/complete.json')):
            d=json.loads(p.read_text());b=d['best'];post=d['post_prune'];s=d['selection']
            row={'arm':arm.name,'alpha':d['alpha'],'cap':d['cap'],'iteration':d['iteration'],
                'accuracy':b['ACC'],'eo':b['EO'],'post_prune_accuracy':post['ACC'],
                'post_prune_eo':post['EO'],'best_epoch':d['best_epoch'],
                'target_sparsity':d['after']['target_sparsity'],
                'model_sparsity':d['after']['model_sparsity'],
                'budget_deficit':s['deficit'],'elapsed_seconds':d['elapsed_seconds'],
                'features1_active':next(l['active_after'] for l in s['layers'] if l['block']=='features.1'),
                'checkpoint':str(p.parent/'retrained.pt')}
            for cell in b['cells']:row[f"species{cell['species']}_color{cell['color']}"]=cell['accuracy']
            for layer in s['layers']:
                if layer['limit'] is not None:assert layer['removed']<=layer['limit']
            assert s['no_cap_relaxation']
            row['max_block_remove_fraction']=max(l['removed_fraction'] for l in s['layers'])
            row['features8_active']=next(l['active_after'] for l in s['layers'] if l['block']=='features.8')
            for layer in s['layers']:
                layer_rows.append({'arm':arm.name,'alpha':d['alpha'],'cap':d['cap'],'iteration':d['iteration'],**layer})
            rows.append(row)
            epochs=json.loads((p.parent/'epochs.json').read_text())
            for e in epochs:
                epoch_rows.append({'arm':arm.name,'alpha':d['alpha'],'cap':d['cap'],
                                   'iteration':d['iteration'],'epoch':e['epoch'],
                                   'accuracy':e['accuracy'],'eo':e['eo']})
    for name,data in [('comparison.csv',rows),('epochs_comparison.csv',epoch_rows),('layers_comparison.csv',layer_rows)]:
        if not data:continue
        with (run/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print('alpha cap iteration accuracy EO target_sparsity first_block_channels')
    for row in rows:
        print(row['alpha'],row['cap'],row['iteration'],
              *[round(row[k],6) for k in ['accuracy','eo','target_sparsity']],row['features1_active'])
    if not rows:return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    colors={None:'#64748b',.2:'#2563eb',.1:'#e65f2b'}
    for ai,alpha in enumerate([.9,.7]):
        for cap in [None,.2,.1]:
            data=sorted([r for r in rows if r['alpha']==alpha and r['cap']==cap],key=lambda r:r['iteration'])
            if not data:continue
            initial=json.loads((run/data[0]['arm']/'initial/metrics.json').read_text())
            pref=json.loads((run/data[0]['arm']/'preflight.json').read_text())
            x=[25]+[r['iteration'] for r in data]
            values=[[initial['ACC']]+[r['accuracy'] for r in data],
                    [initial['EO']]+[r['eo'] for r in data],
                    [pref['source_layout']['features.1']['active']]+[r['features1_active'] for r in data]]
            for j,v in enumerate(values):
                axs[ai,j].plot(x,v,marker='o',linestyle='--' if cap==.2 else '-',color=colors[cap],label='OFF' if cap is None else f'cap {cap:.0%}')
        for j,title in enumerate(['Validation accuracy','Validation EO (lower better)','First-block live channels']):
            axs[ai,j].set_title(f'alpha={alpha}: {title}');axs[ai,j].set_xlabel('Pruning iteration')
            axs[ai,j].grid(alpha=.2)
            if axs[ai,j].lines:axs[ai,j].legend(frameon=False)
            axs[ai,j].set_xticks([25,26,27,28])
            if j==2:
                axs[ai,j].set_ylim(0,10);axs[ai,j].set_yticks([0,2,4,6,8,10])
    fig.suptitle('DogsCats beta=10: paired 3-iteration cap smoke runs')
    fig.savefig(run/'comparison.png',dpi=180);fig.savefig(run/'comparison.pdf')
    plt.close(fig)
    # Final live-channel differences expose redistribution beyond the first block.
    import numpy as np
    grid=[];labels=[]
    for alpha in [.9,.7]:
        off={r['block']:r['active_after'] for r in layer_rows if r['alpha']==alpha and r['cap'] is None and r['iteration']==28}
        if len(off)!=17:continue
        for cap in [.2,.1]:
            current={r['block']:r['active_after'] for r in layer_rows if r['alpha']==alpha and r['cap']==cap and r['iteration']==28}
            if len(current)!=17:continue
            grid.append([current[f'features.{i}']-off[f'features.{i}'] for i in range(1,18)])
            labels.append(f'alpha={alpha}, cap {cap:.0%}')
    if grid:
        values=np.array(grid);bound=max(1,int(np.abs(values).max()))
        fig,ax=plt.subplots(figsize=(13,max(2.4,len(grid)*.75)),constrained_layout=True)
        im=ax.imshow(values,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_xticks(range(17));ax.set_xticklabels(range(1,18));ax.set_xlabel('MobileNetV2 block (features.N)')
        ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(17):
                ax.text(j,i,f'{values[i,j]:+d}',ha='center',va='center',color='white' if abs(values[i,j])>.6*bound else 'black')
        ax.set_title('Iteration 28: live-channel count difference from paired OFF (positive = more kept)')
        fig.colorbar(im,ax=ax,label='Live channels versus OFF',shrink=.8)
        fig.savefig(run/'channel_redistribution.png',dpi=180);fig.savefig(run/'channel_redistribution.pdf');plt.close(fig)



    endpoints={(r['alpha'],r['cap']):r for r in rows if r['iteration']==28}
    if all((alpha,cap) in endpoints for alpha in [.9,.7] for cap in [None,.1]):
        shown=[endpoints[alpha,cap] for alpha in [.9,.7] for cap in [None,.1]]
        xmin=min(r['eo'] for r in shown);xmax=max(r['eo'] for r in shown)
        ymin=min(100*r['accuracy'] for r in shown);ymax=max(100*r['accuracy'] for r in shown)
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True,constrained_layout=True)
        for ax,cap,title in zip(axes,[None,.1],['Cap OFF','10% cap']):
            points=[endpoints[alpha,cap] for alpha in [.9,.7]]
            ax.plot([r['eo'] for r in points],[100*r['accuracy'] for r in points],color='#94a3b8',linestyle='--',zorder=1)
            for alpha,color,marker in [(.9,'#2563eb','D'),(.7,'#e65f2b','o')]:
                r=endpoints[alpha,cap];ax.scatter(r['eo'],100*r['accuracy'],color=color,marker=marker,s=90,zorder=2)
                right_aligned = alpha==.9 or cap is None
                offset=(-8,12) if alpha==.9 else ((-8,-29) if right_aligned else (8,-29))
                label=f"alpha={alpha}\nAcc {100*r['accuracy']:.2f}%, EO {r['eo']:.4f}"
                ax.annotate(label,(r['eo'],100*r['accuracy']),xytext=offset,textcoords='offset points',
                            ha='right' if right_aligned else 'left',fontsize=9)
            ax.set_title(title);ax.set_xlabel('Validation EO (lower is better)');ax.grid(alpha=.2)
            ax.set_xlim(xmin-.009,xmax+.009);ax.set_ylim(ymin-.7,ymax+.7)
        axes[0].set_ylabel('Validation accuracy (%)')
        fig.suptitle('DogsCats: alpha trade-off after 3 pruning iterations (25 to 28)\nEach alpha resumes its own iteration-25 checkpoint',fontsize=12)
        fig.savefig(run/'tradeoff.png',dpi=180);fig.savefig(run/'tradeoff.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path)
    summarize(p.parse_args().run_dir.resolve())
