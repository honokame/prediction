#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
font_lavel = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-mincho.ttf")
font_memori = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")


#%%
#全体のパラメータ
plt.rcParams["font.family"] =  "Times New Roman"# フォント
plt.rcParams["xtick.labelsize"] = 25 # x軸のフォントサイズ
plt.rcParams["ytick.labelsize"] = 25  # y軸のフォントサイズ

#%%
# データ読み込み
df = pd.read_csv('/home/honoka/research/prediction/csv/all.csv')
plt.figure(figsize=(15,10),dpi=700)

# 軸ラベル設定
plt.xlabel("経過時間 [t]",size=30,labelpad=15,fontdict={"fontproperties":font_lavel})
plt.ylabel("測定電圧 [V]",size=30,labelpad=15,fontdict={"fontproperties": font_lavel})

# 軸設定
plt.xlim([0, 1.5])
plt.ylim([0,3.0])
plt.xticks([0,0.5,1,1.5])

plt.grid()  # グリッド線表示

# x軸データ
x = df['time']

# y軸データ
plt.plot(x,df['e1'],c='navy',lw=2.5)
plt.plot(x,df['e2'],c='navy',lw=2.5)
plt.plot(x,df['e3'],c='navy',lw=2.5)
plt.plot(x,df['e4'],c='navy',lw=2.5)
plt.plot(x,df['e5'],c='navy',lw=2.5)
plt.plot(x,df['e6'],c='navy',lw=2.5)
plt.plot(x,df['e7'],c='navy',lw=2.5)
plt.plot(x,df['e8'],c='navy',lw=2.5)
plt.plot(x,df['e9'],c='navy',lw=2.5)
plt.plot(x,df['e10'],c='navy',lw=2.5)
plt.plot(x,df['e11'],c='navy',lw=2.5)
plt.plot(x,df['e12'],c='navy',lw=2.5)
plt.plot(x,df['e13'],c='navy',lw=2.5)
plt.plot(x,df['e14'],c='navy',lw=2.5)
plt.plot(x,df['e15'],c='navy',lw=2.5)
plt.plot(x,df['e16'],c='navy',lw=2.5)
plt.plot(x,df['e17'],c='navy',lw=2.5)
plt.plot(x,df['e18'],c='navy',lw=2.5)
plt.plot(x,df['e19'],c='navy',lw=2.5)
plt.plot(x,df['e20'],c='navy',lw=2.5)
plt.plot(x,df['e21'],c='navy',lw=2.5)
plt.plot(x,df['e22'],c='navy',lw=2.5)
plt.plot(x,df['e33'],c='navy',lw=2.5)
plt.plot(x,df['e24'],c='navy',lw=2.5)
plt.plot(x,df['e25'],c='navy',lw=2.5)
plt.plot(x,df['e26'],c='navy',lw=2.5)
plt.plot(x,df['e27'],c='navy',lw=2.5)
plt.plot(x,df['e28'],c='navy',lw=2.5)
plt.plot(x,df['e29'],c='navy',lw=2.5)
plt.plot(x,df['e30'],c='navy',lw=2.5)
plt.plot(x,df['e31'],c='navy',lw=2.5)
plt.plot(x,df['e32'],c='navy',lw=2.5)
plt.plot(x,df['e33'],c='navy',lw=2.5)
plt.plot(x,df['e34'],c='navy',lw=2.5)
plt.plot(x,df['e35'],c='navy',lw=2.5)
plt.plot(x,df['e36'],c='navy',lw=2.5)
plt.plot(x,df['e37'],c='navy',lw=2.5)
plt.plot(x,df['e38'],c='navy',lw=2.5)
plt.plot(x,df['e39'],c='navy',lw=2.5)
plt.plot(x,df['e40'],c='navy',lw=2.5)
plt.plot(x,df['e41'],c='navy',lw=2.5)
plt.plot(x,df['e42'],c='navy',lw=2.5)
plt.plot(x,df['e43'],c='navy',lw=2.5)
plt.plot(x,df['e44'],c='navy',lw=2.5)
plt.plot(x,df['e45'],c='navy',lw=2.5)
plt.plot(x,df['e46'],c='navy',lw=2.5)
plt.plot(x,df['e47'],c='navy',lw=2.5)
plt.plot(x,df['e48'],c='navy',lw=2.5)
plt.plot(x,df['e49'],c='navy',lw=2.5)
plt.plot(x,df['e50'],c='navy',lw=2.5)
plt.plot(x,df['e51'],c='navy',lw=2.5)
plt.plot(x,df['e52'],c='navy',lw=2.5)
plt.plot(x,df['e53'],c='navy',lw=2.5)
plt.plot(x,df['e54'],c='navy',lw=2.5)
plt.plot(x,df['e55'],c='navy',lw=2.5)
plt.plot(x,df['e56'],c='navy',lw=2.5)
plt.plot(x,df['e57'],c='navy',lw=2.5)
plt.plot(x,df['e58'],c='navy',lw=2.5)
plt.plot(x,df['e59'],c='navy',lw=2.5)
plt.plot(x,df['e60'],c='navy',lw=2.5)
plt.plot(x,df['e61'],c='navy',lw=2.5)
plt.plot(x,df['e62'],c='navy',lw=2.5)
plt.plot(x,df['e63'],c='navy',lw=2.5)
plt.plot(x,df['e64'],c='navy',lw=2.5)
plt.plot(x,df['e65'],c='navy',lw=2.5)
plt.plot(x,df['e66'],c='navy',lw=2.5)
plt.plot(x,df['e67'],c='navy',lw=2.5)
plt.plot(x,df['e68'],c='navy',lw=2.5)
plt.plot(x,df['e69'],c='navy',lw=2.5)
plt.plot(x,df['e70'],c='navy',lw=2.5)
plt.plot(x,df['e71'],c='navy',lw=2.5)
plt.plot(x,df['e72'],c='navy',lw=2.5)
plt.plot(x,df['e73'],c='navy',lw=2.5)
plt.plot(x,df['e74'],c='navy',lw=2.5)
plt.plot(x,df['e75'],c='navy',lw=2.5)
plt.plot(x,df['e76'],c='navy',lw=2.5)
plt.plot(x,df['e77'],c='navy',lw=2.5)
plt.plot(x,df['e78'],c='navy',lw=2.5)
plt.plot(x,df['e79'],c='navy',lw=2.5)
plt.plot(x,df['e80'],c='navy',lw=2.5)
plt.plot(x,df['e81'],c='navy',lw=2.5)
plt.plot(x,df['e82'],c='navy',lw=2.5)
plt.plot(x,df['e83'],c='navy',lw=2.5)
plt.plot(x,df['e84'],c='navy',lw=2.5)
plt.plot(x,df['e85'],c='navy',lw=2.5)
plt.plot(x,df['e86'],c='navy',lw=2.5)
plt.plot(x,df['e87'],c='navy',lw=2.5)
plt.plot(x,df['e88'],c='navy',lw=2.5)
plt.plot(x,df['e89'],c='navy',lw=2.5)
plt.plot(x,df['e90'],c='navy',lw=2.5)
plt.plot(x,df['e91'],c='navy',lw=2.5)
plt.plot(x,df['e92'],c='navy',lw=2.5)
plt.plot(x,df['e93'],c='navy',lw=2.5)
plt.plot(x,df['e94'],c='navy',lw=2.5)
plt.plot(x,df['e95'],c='navy',lw=2.5)
plt.plot(x,df['e96'],c='navy',lw=2.5)
plt.plot(x,df['e97'],c='navy',lw=2.5)
plt.plot(x,df['e98'],c='navy',lw=2.5)
plt.plot(x,df['e99'],c='navy',lw=2.5)
plt.plot(x,df['e100'],c='navy',lw=2.5)
plt.plot(x,df['e101'],c='navy',lw=2.5)
plt.plot(x,df['e102'],c='navy',lw=2.5)
plt.plot(x,df['e103'],c='navy',lw=2.5)
plt.plot(x,df['e104'],c='navy',lw=2.5)
plt.plot(x,df['e105'],c='navy',lw=2.5)
plt.plot(x,df['e106'],c='navy',lw=2.5)
plt.plot(x,df['e107'],c='navy',lw=2.5)
plt.plot(x,df['e108'],c='navy',lw=2.5)
plt.plot(x,df['e109'],c='navy',lw=2.5)
plt.plot(x,df['e110'],c='navy',lw=2.5)
plt.plot(x,df['e111'],c='navy',lw=2.5)
plt.plot(x,df['e112'],c='navy',lw=2.5)
plt.plot(x,df['e113'],c='navy',lw=2.5)
plt.plot(x,df['e114'],c='navy',lw=2.5)
plt.plot(x,df['e115'],c='navy',lw=2.5)
plt.plot(x,df['e116'],c='navy',lw=2.5)
plt.plot(x,df['e117'],c='navy',lw=2.5)
plt.plot(x,df['e118'],c='navy',lw=2.5)
plt.plot(x,df['e119'],c='navy',lw=2.5)
plt.plot(x,df['e110'],c='navy',lw=2.5)
plt.plot(x,df['e111'],c='navy',lw=2.5)
plt.plot(x,df['e112'],c='navy',lw=2.5)
plt.plot(x,df['e113'],c='navy',lw=2.5)
plt.plot(x,df['e114'],c='navy',lw=2.5)
plt.plot(x,df['e115'],c='navy',lw=2.5)
plt.plot(x,df['e116'],c='navy',lw=2.5)
plt.plot(x,df['e117'],c='navy',lw=2.5)
plt.plot(x,df['e118'],c='navy',lw=2.5)
plt.plot(x,df['e119'],c='navy',lw=2.5)
plt.plot(x,df['e120'],c='navy',lw=2.5)
plt.plot(x,df['e121'],c='navy',lw=2.5)
plt.plot(x,df['e122'],c='navy',lw=2.5)
plt.plot(x,df['e123'],c='navy',lw=2.5)
plt.plot(x,df['e124'],c='navy',lw=2.5)
plt.plot(x,df['e125'],c='navy',lw=2.5)
plt.plot(x,df['e126'],c='navy',lw=2.5)
plt.plot(x,df['e127'],c='navy',lw=2.5)
plt.plot(x,df['e128'],c='navy',lw=2.5)
plt.plot(x,df['e129'],c='navy',lw=2.5)
plt.plot(x,df['e130'],c='navy',lw=2.5)
plt.plot(x,df['e131'],c='navy',lw=2.5)
plt.plot(x,df['e132'],c='navy',lw=2.5)
plt.plot(x,df['e133'],c='navy',lw=2.5)
plt.plot(x,df['e134'],c='navy',lw=2.5)
plt.plot(x,df['e135'],c='navy',lw=2.5)
plt.plot(x,df['e136'],c='navy',lw=2.5)
plt.plot(x,df['e137'],c='navy',lw=2.5)
plt.plot(x,df['e138'],c='navy',lw=2.5)
plt.plot(x,df['e139'],c='navy',lw=2.5)
plt.plot(x,df['e140'],c='navy',lw=2.5)
plt.plot(x,df['e141'],c='navy',lw=2.5)
plt.plot(x,df['e142'],c='navy',lw=2.5)
plt.plot(x,df['e143'],c='navy',lw=2.5)
plt.plot(x,df['e144'],c='navy',lw=2.5)
plt.plot(x,df['e145'],c='navy',lw=2.5)
plt.plot(x,df['e146'],c='navy',lw=2.5)
plt.plot(x,df['e147'],c='navy',lw=2.5)
plt.plot(x,df['e148'],c='navy',lw=2.5)
plt.plot(x,df['e149'],c='navy',lw=2.5)
plt.plot(x,df['e150'],c='navy',lw=2.5)
plt.plot(x,df['e151'],c='navy',lw=2.5)
plt.plot(x,df['e152'],c='navy',lw=2.5)
plt.plot(x,df['e153'],c='navy',lw=2.5)
plt.plot(x,df['e154'],c='navy',lw=2.5)
plt.plot(x,df['e155'],c='navy',lw=2.5)
plt.plot(x,df['e156'],c='navy',lw=2.5)
plt.plot(x,df['e157'],c='navy',lw=2.5)
plt.plot(x,df['e158'],c='navy',lw=2.5)
plt.plot(x,df['e159'],c='navy',lw=2.5)
plt.plot(x,df['e160'],c='navy',lw=2.5)
plt.plot(x,df['e161'],c='navy',lw=2.5)
plt.plot(x,df['e162'],c='navy',lw=2.5)
plt.plot(x,df['e163'],c='navy',lw=2.5)
plt.plot(x,df['e164'],c='navy',lw=2.5)
plt.plot(x,df['e165'],c='navy',lw=2.5)
plt.plot(x,df['e166'],c='navy',lw=2.5)
plt.plot(x,df['e167'],c='navy',lw=2.5)
plt.plot(x,df['e168'],c='navy',lw=2.5)
plt.plot(x,df['e169'],c='navy',lw=2.5)
plt.plot(x,df['e170'],c='navy',lw=2.5)
plt.plot(x,df['e171'],c='navy',lw=2.5)
plt.plot(x,df['e172'],c='navy',lw=2.5)
plt.plot(x,df['e173'],c='navy',lw=2.5)
plt.plot(x,df['e174'],c='navy',lw=2.5)
plt.plot(x,df['e175'],c='navy',lw=2.5)
plt.plot(x,df['e176'],c='navy',lw=2.5)
plt.plot(x,df['e177'],c='navy',lw=2.5)
plt.plot(x,df['e178'],c='navy',lw=2.5)
plt.plot(x,df['e179'],c='navy',lw=2.5)
plt.plot(x,df['e180'],c='navy',lw=2.5)
plt.plot(x,df['e181'],c='navy',lw=2.5)
plt.plot(x,df['e182'],c='navy',lw=2.5)
plt.plot(x,df['e183'],c='navy',lw=2.5)
plt.plot(x,df['e184'],c='navy',lw=2.5)
plt.plot(x,df['e185'],c='navy',lw=2.5)
plt.plot(x,df['e186'],c='navy',lw=2.5)
plt.plot(x,df['e187'],c='navy',lw=2.5)
plt.plot(x,df['e188'],c='navy',lw=2.5)
plt.plot(x,df['e189'],c='navy',lw=2.5)
plt.plot(x,df['e190'],c='navy',lw=2.5)
plt.plot(x,df['e191'],c='navy',lw=2.5)
plt.plot(x,df['e192'],c='navy',lw=2.5)
plt.plot(x,df['e193'],c='navy',lw=2.5)
plt.plot(x,df['e194'],c='navy',lw=2.5)
plt.plot(x,df['e195'],c='navy',lw=2.5)
plt.plot(x,df['e196'],c='navy',lw=2.5)
plt.plot(x,df['e197'],c='navy',lw=2.5)
plt.plot(x,df['e198'],c='navy',lw=2.5)
plt.plot(x,df['e199'],c='navy',lw=2.5)
plt.plot(x,df['e200'],c='navy',lw=2.5)
#%%
# グラフの保存
plt.savefig('600g.png')
# %%