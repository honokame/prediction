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
plt.ylim([0,2.5])
plt.xticks([0,0.5,1,1.5]) # 軸に表示させる数値

# グリッド線表示
plt.grid()

# x軸データ
x = df['time']

# y軸データ
plt.plot(x,df['d1'],c='orange',lw=2.5)
plt.plot(x,df['d2'],c='orange',lw=2.5)
plt.plot(x,df['d3'],c='orange',lw=2.5)
plt.plot(x,df['d4'],c='orange',lw=2.5)
plt.plot(x,df['d5'],c='orange',lw=2.5)
plt.plot(x,df['d6'],c='orange',lw=2.5)
plt.plot(x,df['d7'],c='orange',lw=2.5)
plt.plot(x,df['d8'],c='orange',lw=2.5)
plt.plot(x,df['d9'],c='orange',lw=2.5)
plt.plot(x,df['d10'],c='orange',lw=2.5)
plt.plot(x,df['d11'],c='orange',lw=2.5)
plt.plot(x,df['d12'],c='orange',lw=2.5)
plt.plot(x,df['d13'],c='orange',lw=2.5)
plt.plot(x,df['d14'],c='orange',lw=2.5)
plt.plot(x,df['d15'],c='orange',lw=2.5)
plt.plot(x,df['d16'],c='orange',lw=2.5)
plt.plot(x,df['d17'],c='orange',lw=2.5)
plt.plot(x,df['d18'],c='orange',lw=2.5)
plt.plot(x,df['d19'],c='orange',lw=2.5)
plt.plot(x,df['d20'],c='orange',lw=2.5)
plt.plot(x,df['d21'],c='orange',lw=2.5)
plt.plot(x,df['d22'],c='orange',lw=2.5)
plt.plot(x,df['d33'],c='orange',lw=2.5)
plt.plot(x,df['d24'],c='orange',lw=2.5)
plt.plot(x,df['d25'],c='orange',lw=2.5)
plt.plot(x,df['d26'],c='orange',lw=2.5)
plt.plot(x,df['d27'],c='orange',lw=2.5)
plt.plot(x,df['d28'],c='orange',lw=2.5)
plt.plot(x,df['d29'],c='orange',lw=2.5)
plt.plot(x,df['d30'],c='orange',lw=2.5)
plt.plot(x,df['d31'],c='orange',lw=2.5)
plt.plot(x,df['d32'],c='orange',lw=2.5)
plt.plot(x,df['d33'],c='orange',lw=2.5)
plt.plot(x,df['d34'],c='orange',lw=2.5)
plt.plot(x,df['d35'],c='orange',lw=2.5)
plt.plot(x,df['d36'],c='orange',lw=2.5)
plt.plot(x,df['d37'],c='orange',lw=2.5)
plt.plot(x,df['d38'],c='orange',lw=2.5)
plt.plot(x,df['d39'],c='orange',lw=2.5)
plt.plot(x,df['d40'],c='orange',lw=2.5)
plt.plot(x,df['d41'],c='orange',lw=2.5)
plt.plot(x,df['d42'],c='orange',lw=2.5)
plt.plot(x,df['d43'],c='orange',lw=2.5)
plt.plot(x,df['d44'],c='orange',lw=2.5)
plt.plot(x,df['d45'],c='orange',lw=2.5)
plt.plot(x,df['d46'],c='orange',lw=2.5)
plt.plot(x,df['d47'],c='orange',lw=2.5)
plt.plot(x,df['d48'],c='orange',lw=2.5)
plt.plot(x,df['d49'],c='orange',lw=2.5)
plt.plot(x,df['d50'],c='orange',lw=2.5)
plt.plot(x,df['d51'],c='orange',lw=2.5)
plt.plot(x,df['d52'],c='orange',lw=2.5)
plt.plot(x,df['d53'],c='orange',lw=2.5)
plt.plot(x,df['d54'],c='orange',lw=2.5)
plt.plot(x,df['d55'],c='orange',lw=2.5)
plt.plot(x,df['d56'],c='orange',lw=2.5)
plt.plot(x,df['d57'],c='orange',lw=2.5)
plt.plot(x,df['d58'],c='orange',lw=2.5)
plt.plot(x,df['d59'],c='orange',lw=2.5)
plt.plot(x,df['d60'],c='orange',lw=2.5)
plt.plot(x,df['d61'],c='orange',lw=2.5)
plt.plot(x,df['d62'],c='orange',lw=2.5)
plt.plot(x,df['d63'],c='orange',lw=2.5)
plt.plot(x,df['d64'],c='orange',lw=2.5)
plt.plot(x,df['d65'],c='orange',lw=2.5)
plt.plot(x,df['d66'],c='orange',lw=2.5)
plt.plot(x,df['d67'],c='orange',lw=2.5)
plt.plot(x,df['d68'],c='orange',lw=2.5)
plt.plot(x,df['d69'],c='orange',lw=2.5)
plt.plot(x,df['d70'],c='orange',lw=2.5)
plt.plot(x,df['d71'],c='orange',lw=2.5)
plt.plot(x,df['d72'],c='orange',lw=2.5)
plt.plot(x,df['d73'],c='orange',lw=2.5)
plt.plot(x,df['d74'],c='orange',lw=2.5)
plt.plot(x,df['d75'],c='orange',lw=2.5)
plt.plot(x,df['d76'],c='orange',lw=2.5)
plt.plot(x,df['d77'],c='orange',lw=2.5)
plt.plot(x,df['d78'],c='orange',lw=2.5)
plt.plot(x,df['d79'],c='orange',lw=2.5)
plt.plot(x,df['d80'],c='orange',lw=2.5)
plt.plot(x,df['d81'],c='orange',lw=2.5)
plt.plot(x,df['d82'],c='orange',lw=2.5)
plt.plot(x,df['d83'],c='orange',lw=2.5)
plt.plot(x,df['d84'],c='orange',lw=2.5)
plt.plot(x,df['d85'],c='orange',lw=2.5)
plt.plot(x,df['d86'],c='orange',lw=2.5)
plt.plot(x,df['d87'],c='orange',lw=2.5)
plt.plot(x,df['d88'],c='orange',lw=2.5)
plt.plot(x,df['d89'],c='orange',lw=2.5)
plt.plot(x,df['d90'],c='orange',lw=2.5)
plt.plot(x,df['d91'],c='orange',lw=2.5)
plt.plot(x,df['d92'],c='orange',lw=2.5)
plt.plot(x,df['d93'],c='orange',lw=2.5)
plt.plot(x,df['d94'],c='orange',lw=2.5)
plt.plot(x,df['d95'],c='orange',lw=2.5)
plt.plot(x,df['d96'],c='orange',lw=2.5)
plt.plot(x,df['d97'],c='orange',lw=2.5)
plt.plot(x,df['d98'],c='orange',lw=2.5)
plt.plot(x,df['d99'],c='orange',lw=2.5)
plt.plot(x,df['d100'],c='orange',lw=2.5)
plt.plot(x,df['d101'],c='orange',lw=2.5)
plt.plot(x,df['d102'],c='orange',lw=2.5)
plt.plot(x,df['d103'],c='orange',lw=2.5)
plt.plot(x,df['d104'],c='orange',lw=2.5)
plt.plot(x,df['d105'],c='orange',lw=2.5)
plt.plot(x,df['d106'],c='orange',lw=2.5)
plt.plot(x,df['d107'],c='orange',lw=2.5)
plt.plot(x,df['d108'],c='orange',lw=2.5)
plt.plot(x,df['d109'],c='orange',lw=2.5)
plt.plot(x,df['d110'],c='orange',lw=2.5)
plt.plot(x,df['d111'],c='orange',lw=2.5)
plt.plot(x,df['d112'],c='orange',lw=2.5)
plt.plot(x,df['d113'],c='orange',lw=2.5)
plt.plot(x,df['d114'],c='orange',lw=2.5)
plt.plot(x,df['d115'],c='orange',lw=2.5)
plt.plot(x,df['d116'],c='orange',lw=2.5)
plt.plot(x,df['d117'],c='orange',lw=2.5)
plt.plot(x,df['d118'],c='orange',lw=2.5)
plt.plot(x,df['d119'],c='orange',lw=2.5)
plt.plot(x,df['d110'],c='orange',lw=2.5)
plt.plot(x,df['d111'],c='orange',lw=2.5)
plt.plot(x,df['d112'],c='orange',lw=2.5)
plt.plot(x,df['d113'],c='orange',lw=2.5)
plt.plot(x,df['d114'],c='orange',lw=2.5)
plt.plot(x,df['d115'],c='orange',lw=2.5)
plt.plot(x,df['d116'],c='orange',lw=2.5)
plt.plot(x,df['d117'],c='orange',lw=2.5)
plt.plot(x,df['d118'],c='orange',lw=2.5)
plt.plot(x,df['d119'],c='orange',lw=2.5)
plt.plot(x,df['d120'],c='orange',lw=2.5)
plt.plot(x,df['d121'],c='orange',lw=2.5)
plt.plot(x,df['d122'],c='orange',lw=2.5)
plt.plot(x,df['d123'],c='orange',lw=2.5)
plt.plot(x,df['d124'],c='orange',lw=2.5)
plt.plot(x,df['d125'],c='orange',lw=2.5)
plt.plot(x,df['d126'],c='orange',lw=2.5)
plt.plot(x,df['d127'],c='orange',lw=2.5)
plt.plot(x,df['d128'],c='orange',lw=2.5)
plt.plot(x,df['d129'],c='orange',lw=2.5)
plt.plot(x,df['d130'],c='orange',lw=2.5)
plt.plot(x,df['d131'],c='orange',lw=2.5)
plt.plot(x,df['d132'],c='orange',lw=2.5)
plt.plot(x,df['d133'],c='orange',lw=2.5)
plt.plot(x,df['d134'],c='orange',lw=2.5)
plt.plot(x,df['d135'],c='orange',lw=2.5)
plt.plot(x,df['d136'],c='orange',lw=2.5)
plt.plot(x,df['d137'],c='orange',lw=2.5)
plt.plot(x,df['d138'],c='orange',lw=2.5)
plt.plot(x,df['d139'],c='orange',lw=2.5)
plt.plot(x,df['d140'],c='orange',lw=2.5)
plt.plot(x,df['d141'],c='orange',lw=2.5)
plt.plot(x,df['d142'],c='orange',lw=2.5)
plt.plot(x,df['d143'],c='orange',lw=2.5)
plt.plot(x,df['d144'],c='orange',lw=2.5)
plt.plot(x,df['d145'],c='orange',lw=2.5)
plt.plot(x,df['d146'],c='orange',lw=2.5)
plt.plot(x,df['d147'],c='orange',lw=2.5)
plt.plot(x,df['d148'],c='orange',lw=2.5)
plt.plot(x,df['d149'],c='orange',lw=2.5)
plt.plot(x,df['d150'],c='orange',lw=2.5)
plt.plot(x,df['d151'],c='orange',lw=2.5)
plt.plot(x,df['d152'],c='orange',lw=2.5)
plt.plot(x,df['d153'],c='orange',lw=2.5)
plt.plot(x,df['d154'],c='orange',lw=2.5)
plt.plot(x,df['d155'],c='orange',lw=2.5)
plt.plot(x,df['d156'],c='orange',lw=2.5)
plt.plot(x,df['d157'],c='orange',lw=2.5)
plt.plot(x,df['d158'],c='orange',lw=2.5)
plt.plot(x,df['d159'],c='orange',lw=2.5)
plt.plot(x,df['d160'],c='orange',lw=2.5)
plt.plot(x,df['d161'],c='orange',lw=2.5)
plt.plot(x,df['d162'],c='orange',lw=2.5)
plt.plot(x,df['d163'],c='orange',lw=2.5)
plt.plot(x,df['d164'],c='orange',lw=2.5)
plt.plot(x,df['d165'],c='orange',lw=2.5)
plt.plot(x,df['d166'],c='orange',lw=2.5)
plt.plot(x,df['d167'],c='orange',lw=2.5)
plt.plot(x,df['d168'],c='orange',lw=2.5)
plt.plot(x,df['d169'],c='orange',lw=2.5)
plt.plot(x,df['d170'],c='orange',lw=2.5)
plt.plot(x,df['d171'],c='orange',lw=2.5)
plt.plot(x,df['d172'],c='orange',lw=2.5)
plt.plot(x,df['d173'],c='orange',lw=2.5)
plt.plot(x,df['d174'],c='orange',lw=2.5)
plt.plot(x,df['d175'],c='orange',lw=2.5)
plt.plot(x,df['d176'],c='orange',lw=2.5)
plt.plot(x,df['d177'],c='orange',lw=2.5)
plt.plot(x,df['d178'],c='orange',lw=2.5)
plt.plot(x,df['d179'],c='orange',lw=2.5)
plt.plot(x,df['d180'],c='orange',lw=2.5)
plt.plot(x,df['d181'],c='orange',lw=2.5)
plt.plot(x,df['d182'],c='orange',lw=2.5)
plt.plot(x,df['d183'],c='orange',lw=2.5)
plt.plot(x,df['d184'],c='orange',lw=2.5)
plt.plot(x,df['d185'],c='orange',lw=2.5)
plt.plot(x,df['d186'],c='orange',lw=2.5)
plt.plot(x,df['d187'],c='orange',lw=2.5)
plt.plot(x,df['d188'],c='orange',lw=2.5)
plt.plot(x,df['d189'],c='orange',lw=2.5)
plt.plot(x,df['d190'],c='orange',lw=2.5)
plt.plot(x,df['d191'],c='orange',lw=2.5)
plt.plot(x,df['d192'],c='orange',lw=2.5)
plt.plot(x,df['d193'],c='orange',lw=2.5)
plt.plot(x,df['d194'],c='orange',lw=2.5)
plt.plot(x,df['d195'],c='orange',lw=2.5)
plt.plot(x,df['d196'],c='orange',lw=2.5)
plt.plot(x,df['d197'],c='orange',lw=2.5)
plt.plot(x,df['d198'],c='orange',lw=2.5)
plt.plot(x,df['d199'],c='orange',lw=2.5)
plt.plot(x,df['d200'],c='orange',lw=2.5)
#%%
# グラフの保存
plt.savefig('/home/honoka/research/prediction/csv/image/500g.png')