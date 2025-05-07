import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('./mae.xlsx')

# 查找最低点
min_mae_index = df['mae'].idxmin()
min_mae_value = df['mae'].min()

plt.figure(figsize=(10,6))
plt.plot(range(1,len(df)+1),df["mae"],color='b',linestyle='-')
plt.title('MAE Over Time')
plt.xlabel('Index')
plt.ylabel('MAE')

plt.plot(min_mae_index+1,min_mae_value,marker='o',color='r')
plt.annotate(f'Index: {min_mae_index + 1}  MAE: {min_mae_value:.2f}',
             xy=(min_mae_index + 1, min_mae_value),
             xytext=(min_mae_index + 1, min_mae_value + 3),
             arrowprops=dict(facecolor='red', shrink=0.01),
             horizontalalignment='left',
             verticalalignment='bottom')
plt.savefig('./出图/result/mae_vis_city.png', format='png', dpi=300)
plt.show()