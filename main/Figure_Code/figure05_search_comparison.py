"""Correct Figure 5 data using its original plotting layout, without changing other figures."""
from pathlib import Path
import ast
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'tmp/v45_coefficient_scope_20260918'
sys.path.insert(0,str(QA/'dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

OUT=ROOT/'figure_revision_preview/v45_coefficient_correction_20260918'
OUT.mkdir(parents=True,exist_ok=True)
for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    p=ROOT/'Times new Roman'/name
    if p.exists(): font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','axes.linewidth':0.9,'savefig.dpi':300,
                     'svg.fonttype':'path','svg.hashsalt':'v45-coefficient-correction'})

def main():
    evidence=json.loads((QA/'coefficient_evidence.json').read_text(encoding='utf-8'))
    layer=next(x['coefficient_pct'] for x in evidence['records'] if x['model']=='8B' and x['mode']=='layer')
    glob=next(x['coefficient_pct'] for x in evidence['records'] if x['model']=='8B' and x['mode']=='global')
    original=ROOT/'论文修订v05/新图表/build_data_region_figures.py'
    source=original.read_text(encoding='utf-8')
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fig_search_compare')
    function=ast.get_source_segment(source,fn)
    function=function.replace('("Base weight share (%)",    "lower is better",  49.2, 26.3, None)',
                              f'("Mean SLERP coeff. (%)",    "layer - global",  {glob!r}, {layer!r}, None)')
    assert '26.3' not in function
    # Values nearly coincide: show the two existing circle markers at a small vertical offset.
    function=function.replace('ax.plot([lo, hi], [y, y],',
        'yg, yp = (y + 0.09, y - 0.09) if y == 0 else (y, y)\n        ax.plot([g, p], [yg, yp],')
    function=function.replace('ax.scatter([g], [y]', 'ax.scatter([g], [yg]')
    function=function.replace('ax.scatter([p], [y]', 'ax.scatter([p], [yp]')
    function=function.replace('ax.text(g - OFF, y,','ax.text(g - OFF, yg,')
    function=function.replace('ax.text(p + OFF, y,','ax.text(p + OFF, yp,')
    function=function.replace('ax.text(g + OFF, y,','ax.text(g + OFF, yg,')
    function=function.replace('ax.text(p - OFF, y,','ax.text(p - OFF, yp,')
    # Keep the existing canvas, axes, three rows, colors, fonts, and the two evaluation rows.
    ns={'plt':plt,'Line2D':__import__('matplotlib.lines',fromlist=['Line2D']).Line2D,
        'ACCENT':'#0e7c66','MID_SLATE':'#94a3b8','REF_RED':'#dc2626',
        'INK':'#1f2937','MUTE':'#6b7280'}
    def save(fig,stem):
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        tight=fig.get_tightbbox(fig.canvas.get_renderer())
        width,height=2397/300,1101/300
        assert tight.width < width and tight.height < height
        box=Bbox.from_bounds(tight.x0-(width-tight.width)/2,tight.y0-(height-tight.height)/2,width,height)
        fig.savefig(OUT/'Figure05_corrected.svg',bbox_inches=box,
                    facecolor='white',metadata={'Date':None})
        fig.savefig(OUT/'Figure05_corrected.pdf',bbox_inches=box,
                    facecolor='white',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(OUT/'Figure05_corrected.png',bbox_inches=box,facecolor='white')
        plt.close(fig)
        print('Figure5 pixels:',Image.open(OUT/'Figure05_corrected.png').size)
    ns['save']=save
    exec(compile(function,str(original)+'::scoped_figure5','exec'),ns)
    ns['fig_search_compare']()
    (OUT/'data.json').write_text(json.dumps({'8B_global_mean_coefficient_pct':glob,
        '8B_layer_mean_coefficient_pct':layer,'difference_pp':layer-glob,
        'ASR_percent':[52.7,55.3],'DQR_percent':[73.4,80.0],
        'source_plot':str(original),'new_label':'Mean SLERP coeff. (%)',
        'near_coincident_markers':'global +0.09 and layer -0.09 y-offset, horizontal positions remain true'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
