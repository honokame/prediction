#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
font_lavel = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/fonts-japanese-mincho.ttf")
#%%
#全体のパラメータ
plt.rcParams["font.family"] =  "Times New Roman" # フォント
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
plt.xlim([0, 1.5]) # 軸範囲
plt.ylim([0,3.5])
plt.xticks([0,0.5,1,1.5]) # 軸に表示させる数値

グリッド線表示
plt.grid()

# x軸データ
x = df['time']

# y軸データ
plt.plot(x,df['f1'],c='limegreen',lw=2.5)
plt.plot(x,df['f2'],c='limegreen',lw=2.5)
plt.plot(x,df['f3'],c='limegreen',lw=2.5)
plt.plot(x,df['f4'],c='limegreen',lw=2.5)
plt.plot(x,df['f5'],c='limegreen',lw=2.5)
plt.plot(x,df['f6'],c='limegreen',lw=2.5)
plt.plot(x,df['f7'],c='limegreen',lw=2.5)
plt.plot(x,df['f8'],c='limegreen',lw=2.5)
plt.plot(x,df['f9'],c='limegreen',lw=2.5)
plt.plot(x,df['f10'],c='limegreen',lw=2.5)
plt.plot(x,df['f11'],c='limegreen',lw=2.5)
plt.plot(x,df['f12'],c='limegreen',lw=2.5)
plt.plot(x,df['f13'],c='limegreen',lw=2.5)
plt.plot(x,df['f14'],c='limegreen',lw=2.5)
plt.plot(x,df['f15'],c='limegreen',lw=2.5)
plt.plot(x,df['f16'],c='limegreen',lw=2.5)
plt.plot(x,df['f17'],c='limegreen',lw=2.5)
plt.plot(x,df['f18'],c='limegreen',lw=2.5)
plt.plot(x,df['f19'],c='limegreen',lw=2.5)
plt.plot(x,df['f20'],c='limegreen',lw=2.5)
plt.plot(x,df['f21'],c='limegreen',lw=2.5)
plt.plot(x,df['f22'],c='limegreen',lw=2.5)
plt.plot(x,df['f33'],c='limegreen',lw=2.5)
plt.plot(x,df['f24'],c='limegreen',lw=2.5)
plt.plot(x,df['f25'],c='limegreen',lw=2.5)
plt.plot(x,df['f26'],c='limegreen',lw=2.5)
plt.plot(x,df['f27'],c='limegreen',lw=2.5)
plt.plot(x,df['f28'],c='limegreen',lw=2.5)
plt.plot(x,df['f29'],c='limegreen',lw=2.5)
plt.plot(x,df['f30'],c='limegreen',lw=2.5)
plt.plot(x,df['f31'],c='limegreen',lw=2.5)
plt.plot(x,df['f32'],c='limegreen',lw=2.5)
plt.plot(x,df['f33'],c='limegreen',lw=2.5)
plt.plot(x,df['f34'],c='limegreen',lw=2.5)
plt.plot(x,df['f35'],c='limegreen',lw=2.5)
plt.plot(x,df['f36'],c='limegreen',lw=2.5)
plt.plot(x,df['f37'],c='limegreen',lw=2.5)
plt.plot(x,df['f38'],c='limegreen',lw=2.5)
plt.plot(x,df['f39'],c='limegreen',lw=2.5)
plt.plot(x,df['f40'],c='limegreen',lw=2.5)
plt.plot(x,df['f41'],c='limegreen',lw=2.5)
plt.plot(x,df['f42'],c='limegreen',lw=2.5)
plt.plot(x,df['f43'],c='limegreen',lw=2.5)
plt.plot(x,df['f44'],c='limegreen',lw=2.5)
plt.plot(x,df['f45'],c='limegreen',lw=2.5)
plt.plot(x,df['f46'],c='limegreen',lw=2.5)
plt.plot(x,df['f47'],c='limegreen',lw=2.5)
plt.plot(x,df['f48'],c='limegreen',lw=2.5)
plt.plot(x,df['f49'],c='limegreen',lw=2.5)
plt.plot(x,df['f50'],c='limegreen',lw=2.5)
plt.plot(x,df['f51'],c='limegreen',lw=2.5)
plt.plot(x,df['f52'],c='limegreen',lw=2.5)
plt.plot(x,df['f53'],c='limegreen',lw=2.5)
plt.plot(x,df['f54'],c='limegreen',lw=2.5)
plt.plot(x,df['f55'],c='limegreen',lw=2.5)
plt.plot(x,df['f56'],c='limegreen',lw=2.5)
plt.plot(x,df['f57'],c='limegreen',lw=2.5)
plt.plot(x,df['f58'],c='limegreen',lw=2.5)
plt.plot(x,df['f59'],c='limegreen',lw=2.5)
plt.plot(x,df['f60'],c='limegreen',lw=2.5)
plt.plot(x,df['f61'],c='limegreen',lw=2.5)
plt.plot(x,df['f62'],c='limegreen',lw=2.5)
plt.plot(x,df['f63'],c='limegreen',lw=2.5)
plt.plot(x,df['f64'],c='limegreen',lw=2.5)
plt.plot(x,df['f65'],c='limegreen',lw=2.5)
plt.plot(x,df['f66'],c='limegreen',lw=2.5)
plt.plot(x,df['f67'],c='limegreen',lw=2.5)
plt.plot(x,df['f68'],c='limegreen',lw=2.5)
plt.plot(x,df['f69'],c='limegreen',lw=2.5)
plt.plot(x,df['f70'],c='limegreen',lw=2.5)
plt.plot(x,df['f71'],c='limegreen',lw=2.5)
plt.plot(x,df['f72'],c='limegreen',lw=2.5)
plt.plot(x,df['f73'],c='limegreen',lw=2.5)
plt.plot(x,df['f74'],c='limegreen',lw=2.5)
plt.plot(x,df['f75'],c='limegreen',lw=2.5)
plt.plot(x,df['f76'],c='limegreen',lw=2.5)
plt.plot(x,df['f77'],c='limegreen',lw=2.5)
plt.plot(x,df['f78'],c='limegreen',lw=2.5)
plt.plot(x,df['f79'],c='limegreen',lw=2.5)
plt.plot(x,df['f80'],c='limegreen',lw=2.5)
plt.plot(x,df['f81'],c='limegreen',lw=2.5)
plt.plot(x,df['f82'],c='limegreen',lw=2.5)
plt.plot(x,df['f83'],c='limegreen',lw=2.5)
plt.plot(x,df['f84'],c='limegreen',lw=2.5)
plt.plot(x,df['f85'],c='limegreen',lw=2.5)
plt.plot(x,df['f86'],c='limegreen',lw=2.5)
plt.plot(x,df['f87'],c='limegreen',lw=2.5)
plt.plot(x,df['f88'],c='limegreen',lw=2.5)
plt.plot(x,df['f89'],c='limegreen',lw=2.5)
plt.plot(x,df['f90'],c='limegreen',lw=2.5)
plt.plot(x,df['f91'],c='limegreen',lw=2.5)
plt.plot(x,df['f92'],c='limegreen',lw=2.5)
plt.plot(x,df['f93'],c='limegreen',lw=2.5)
plt.plot(x,df['f94'],c='limegreen',lw=2.5)
plt.plot(x,df['f95'],c='limegreen',lw=2.5)
plt.plot(x,df['f96'],c='limegreen',lw=2.5)
plt.plot(x,df['f97'],c='limegreen',lw=2.5)
plt.plot(x,df['f98'],c='limegreen',lw=2.5)
plt.plot(x,df['f99'],c='limegreen',lw=2.5)
plt.plot(x,df['f100'],c='limegreen',lw=2.5)
plt.plot(x,df['f101'],c='limegreen',lw=2.5)
plt.plot(x,df['f102'],c='limegreen',lw=2.5)
plt.plot(x,df['f103'],c='limegreen',lw=2.5)
plt.plot(x,df['f104'],c='limegreen',lw=2.5)
plt.plot(x,df['f105'],c='limegreen',lw=2.5)
plt.plot(x,df['f106'],c='limegreen',lw=2.5)
plt.plot(x,df['f107'],c='limegreen',lw=2.5)
plt.plot(x,df['f108'],c='limegreen',lw=2.5)
plt.plot(x,df['f109'],c='limegreen',lw=2.5)
plt.plot(x,df['f110'],c='limegreen',lw=2.5)
plt.plot(x,df['f111'],c='limegreen',lw=2.5)
plt.plot(x,df['f112'],c='limegreen',lw=2.5)
plt.plot(x,df['f113'],c='limegreen',lw=2.5)
plt.plot(x,df['f114'],c='limegreen',lw=2.5)
plt.plot(x,df['f115'],c='limegreen',lw=2.5)
plt.plot(x,df['f116'],c='limegreen',lw=2.5)
plt.plot(x,df['f117'],c='limegreen',lw=2.5)
plt.plot(x,df['f118'],c='limegreen',lw=2.5)
plt.plot(x,df['f119'],c='limegreen',lw=2.5)
plt.plot(x,df['f110'],c='limegreen',lw=2.5)
plt.plot(x,df['f111'],c='limegreen',lw=2.5)
plt.plot(x,df['f112'],c='limegreen',lw=2.5)
plt.plot(x,df['f113'],c='limegreen',lw=2.5)
plt.plot(x,df['f114'],c='limegreen',lw=2.5)
plt.plot(x,df['f115'],c='limegreen',lw=2.5)
plt.plot(x,df['f116'],c='limegreen',lw=2.5)
plt.plot(x,df['f117'],c='limegreen',lw=2.5)
plt.plot(x,df['f118'],c='limegreen',lw=2.5)
plt.plot(x,df['f119'],c='limegreen',lw=2.5)
plt.plot(x,df['f120'],c='limegreen',lw=2.5)
plt.plot(x,df['f121'],c='limegreen',lw=2.5)
plt.plot(x,df['f122'],c='limegreen',lw=2.5)
plt.plot(x,df['f123'],c='limegreen',lw=2.5)
plt.plot(x,df['f124'],c='limegreen',lw=2.5)
plt.plot(x,df['f125'],c='limegreen',lw=2.5)
plt.plot(x,df['f126'],c='limegreen',lw=2.5)
plt.plot(x,df['f127'],c='limegreen',lw=2.5)
plt.plot(x,df['f128'],c='limegreen',lw=2.5)
plt.plot(x,df['f129'],c='limegreen',lw=2.5)
plt.plot(x,df['f130'],c='limegreen',lw=2.5)
plt.plot(x,df['f131'],c='limegreen',lw=2.5)
plt.plot(x,df['f132'],c='limegreen',lw=2.5)
plt.plot(x,df['f133'],c='limegreen',lw=2.5)
plt.plot(x,df['f134'],c='limegreen',lw=2.5)
plt.plot(x,df['f135'],c='limegreen',lw=2.5)
plt.plot(x,df['f136'],c='limegreen',lw=2.5)
plt.plot(x,df['f137'],c='limegreen',lw=2.5)
plt.plot(x,df['f138'],c='limegreen',lw=2.5)
plt.plot(x,df['f139'],c='limegreen',lw=2.5)
plt.plot(x,df['f140'],c='limegreen',lw=2.5)
plt.plot(x,df['f141'],c='limegreen',lw=2.5)
plt.plot(x,df['f142'],c='limegreen',lw=2.5)
plt.plot(x,df['f143'],c='limegreen',lw=2.5)
plt.plot(x,df['f144'],c='limegreen',lw=2.5)
plt.plot(x,df['f145'],c='limegreen',lw=2.5)
plt.plot(x,df['f146'],c='limegreen',lw=2.5)
plt.plot(x,df['f147'],c='limegreen',lw=2.5)
plt.plot(x,df['f148'],c='limegreen',lw=2.5)
plt.plot(x,df['f149'],c='limegreen',lw=2.5)
plt.plot(x,df['f150'],c='limegreen',lw=2.5)
plt.plot(x,df['f151'],c='limegreen',lw=2.5)
plt.plot(x,df['f152'],c='limegreen',lw=2.5)
plt.plot(x,df['f153'],c='limegreen',lw=2.5)
plt.plot(x,df['f154'],c='limegreen',lw=2.5)
plt.plot(x,df['f155'],c='limegreen',lw=2.5)
plt.plot(x,df['f156'],c='limegreen',lw=2.5)
plt.plot(x,df['f157'],c='limegreen',lw=2.5)
plt.plot(x,df['f158'],c='limegreen',lw=2.5)
plt.plot(x,df['f159'],c='limegreen',lw=2.5)
plt.plot(x,df['f160'],c='limegreen',lw=2.5)
plt.plot(x,df['f161'],c='limegreen',lw=2.5)
plt.plot(x,df['f162'],c='limegreen',lw=2.5)
plt.plot(x,df['f163'],c='limegreen',lw=2.5)
plt.plot(x,df['f164'],c='limegreen',lw=2.5)
plt.plot(x,df['f165'],c='limegreen',lw=2.5)
plt.plot(x,df['f166'],c='limegreen',lw=2.5)
plt.plot(x,df['f167'],c='limegreen',lw=2.5)
plt.plot(x,df['f168'],c='limegreen',lw=2.5)
plt.plot(x,df['f169'],c='limegreen',lw=2.5)
plt.plot(x,df['f170'],c='limegreen',lw=2.5)
plt.plot(x,df['f171'],c='limegreen',lw=2.5)
plt.plot(x,df['f172'],c='limegreen',lw=2.5)
plt.plot(x,df['f173'],c='limegreen',lw=2.5)
plt.plot(x,df['f174'],c='limegreen',lw=2.5)
plt.plot(x,df['f175'],c='limegreen',lw=2.5)
plt.plot(x,df['f176'],c='limegreen',lw=2.5)
plt.plot(x,df['f177'],c='limegreen',lw=2.5)
plt.plot(x,df['f178'],c='limegreen',lw=2.5)
plt.plot(x,df['f179'],c='limegreen',lw=2.5)
plt.plot(x,df['f180'],c='limegreen',lw=2.5)
plt.plot(x,df['f181'],c='limegreen',lw=2.5)
plt.plot(x,df['f182'],c='limegreen',lw=2.5)
plt.plot(x,df['f183'],c='limegreen',lw=2.5)
plt.plot(x,df['f184'],c='limegreen',lw=2.5)
plt.plot(x,df['f185'],c='limegreen',lw=2.5)
plt.plot(x,df['f186'],c='limegreen',lw=2.5)
plt.plot(x,df['f187'],c='limegreen',lw=2.5)
plt.plot(x,df['f188'],c='limegreen',lw=2.5)
plt.plot(x,df['f189'],c='limegreen',lw=2.5)
plt.plot(x,df['f190'],c='limegreen',lw=2.5)
plt.plot(x,df['f191'],c='limegreen',lw=2.5)
plt.plot(x,df['f192'],c='limegreen',lw=2.5)
plt.plot(x,df['f193'],c='limegreen',lw=2.5)
plt.plot(x,df['f194'],c='limegreen',lw=2.5)
plt.plot(x,df['f195'],c='limegreen',lw=2.5)
plt.plot(x,df['f196'],c='limegreen',lw=2.5)
plt.plot(x,df['f197'],c='limegreen',lw=2.5)
plt.plot(x,df['f198'],c='limegreen',lw=2.5)
plt.plot(x,df['f199'],c='limegreen',lw=2.5)
plt.plot(x,df['f200'],c='limegreen',lw=2.5)
#%%
# グラフの保存
plt.savefig('/home/honoka/research/prediction/csv/image/700g.png')