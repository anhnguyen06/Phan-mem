#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


#Các thư viện cần thiết
import numpy as np
import pandas as pd
import math as math
from ipywidgets import interact
import ipywidgets as ipw
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.io import show, output_notebook, push_notebook

import time

output_notebook()


# In[2]:


# kích thước vật đúc
L_vatduc = 0.078
R_vatduc = 0.038
# đỉnh 
L1 = 0.028 
R1= 0.038 
# giua
L2 = 0.009
R2 = 0.029
#giua 
L3 = 0.019
R3 = 0.019
#duoi 
L4 = 0.022
R4 = 0.009
# kích thước khuôn
L_khuon = 0.118
R_khuon = 0.052


# In[3]:


#khuôn đúc bằng thép
K1 = 54.45 # [w/mk]  Hệ số dẫn nhiệt 
Cp1= 510 #[j/kg.k]  Nhiệt dung riêng 
p1 = 7500 #[Kg/m^3] Khối lượng riêng 

# Vật đúc nhôm
# Nhôm lỏng
K2 = 108 # [w/mk] Hệ số dẫn nhiệt 
Cp2 = 1226 #[j/kg.k] Nhiệt dung riêng 
p2 = 2614 # [Kg/m^3] Khối lượng riêng

 #nhôm đặc 
K3 = 221.6 # [w/mk] Hệ số dẫn nhiệt 
Cp3 = 876 #[j/kg.k]  Nhiệt dung riêng 
p3 = 2966 # [Kg/m^3] Khối lượng riêng 

# Hệ số dẫn nhiệt tại ranh giới khuôn và vật đúc 
# Hệ số tỏa nhiệt ra môi trường 
h0 =120    
h1 = 160
L=392


# In[4]:


T1 = 750 # nhiệt độ ban đầu vật đúc
T_kettinh = 580 # nhiệt độ kết tinh vật đúc
T_khuôn = 120 # nhiệt độ ban đầu của khuôn thép
T_env = 30 # nhiệt độ môi trường


# In[5]:


dx = 0.001
dy = 0.001
# Số nút tính toán theo chiều x,y
nx = int (R_khuon/(dx))
ny = int (L_khuon/(dy))
nx_vatduc = int (R_vatduc/(dx))
ny_vatduc = int (L_vatduc/(dx))
nx1 = int (R1/(dx))
ny1 = int (L1/(dy))
nx2 = int (R2/(dx))
ny2 = int (L2/(dy))
nx3 = int (R3/(dx))
ny3 = int (L3/(dy))
nx4 = int (R4/(dx))
ny4 = int (L4/(dy))
# Hệ số khuếch tán
alpha1 = K1/(Cp1*p1) # hệ số khuếch tán khuôn
alpha2 = K2/(Cp2*p2) #  hệ số khuếch tán vật đúc lỏng
alpha3 = K3/(Cp3*p3) # hệ số khuếch tán vật đúc rắn
Bio1 = (h1*dx)/K1 
Bio2 = (h0*dx)/K2
dt_ondinh = dx**2/(4*alpha2) #thoi gian on dinh
print('nx :', nx)
print('ny :', ny)
print('nx_vatduc :', nx_vatduc)
print('ny_vatduc :', ny_vatduc)
print('nx1 :', nx1)
print('ny1 :', ny1)
print('nx2 :', nx2)
print('ny2 :', ny2)
print('nx3 :', nx3)
print('ny3 :', ny3)
print('nx4 :', nx4)
print('ny4 :', ny4)
print('alpha1 :', alpha1)
print('alpha2:', alpha2)
print('alpha3:', alpha3)
print('Bio1 :', Bio1)
print('Bio2 :', Bio2)
print('Thời gian ổn định cần thiết:', dt_ondinh)


# In[6]:


# Chọn lại dt <= dt_ondinh
dt=math.pow(10,math.floor(math.log10(dt_ondinh)))
dt


# In[7]:


# Tính Fourier 
F1 = (alpha1*dt)/dx**2
F2 = (alpha2*dt)/dx**2
F3 = (alpha3*dt)/dx**2
print(' Khuon :', F1)
print(' vat duc long :', F2)
print(' vat duc ran :', F3)


# In[8]:


#thoi gian tinh toan 
t = 40
step = int(t/dt) 
step


# ## tạo các mảng 
# 

# In[9]:


T = np.zeros((int(step), int(ny), int(nx)))+ T_khuôn
nx_trungtam = (nx)/2
ny_trungtam = (ny)/2
x_bd1 = int( nx_trungtam - nx1/2)
x_bd2 = int(nx_trungtam - nx2/2)
x_bd3 =  int(nx_trungtam - nx3/2)
x_bd4 =  int(nx_trungtam - nx4/2)
x_kt1 = int(nx_trungtam+1+ nx1/2)
x_kt2 = int(nx_trungtam+1+ nx2/2)
x_kt3 = int(nx_trungtam+1+ nx3/2)
x_kt4 = int(nx_trungtam+1 +nx4/2)
y_bd2 = ny1
y_bd3= y_bd2 + ny2
y_bd4 = y_bd3 + ny3
y_kt = ny_vatduc
print(x_bd1,x_bd2,x_bd3,x_bd4,x_kt1,x_kt2,x_kt3,x_kt4)
print(y_bd2,y_bd3,y_bd4,y_kt)
print(T.shape)  


# In[10]:


T[:,0:y_bd2,x_bd1:x_kt1-1] = T1 
T[:,y_bd2:y_bd3,x_bd2:x_kt2] = T1 
T[:,y_bd3:y_bd4,x_bd3:x_kt3] = T1
T[:,y_bd4:y_kt+1,x_bd4:x_kt4] = T1 


# In[11]:


VL = np.zeros((int(step), int(ny), int(nx))) #vì chạy từ 0 nên chỉ cần +1 nút giao biên và 2 gia biên cho y,x
VL[:,0:y_bd2,x_bd1:x_kt1-1] = 1 
VL[:,y_bd2:y_bd3,x_bd2:x_kt2] = 1 
VL[:,y_bd3:y_bd4,x_bd3:x_kt3] = 1
VL[:,y_bd4:y_kt+1,x_bd4:x_kt4] = 1 


# In[12]:


F = np.zeros((int(step), int(ny), int(nx))) +F1 
F[:,0:y_bd2,x_bd1:x_kt1-1] = F2 
F[:,y_bd2:y_bd3,x_bd2:x_kt2] = F2 
F[:,y_bd3:y_bd4,x_bd3:x_kt3] = F2
F[:,y_bd4:y_kt+1,x_bd4:x_kt4] = F2 


# In[13]:


KQ = np.zeros((int(step), int(ny), int(nx)))
KQ[:,0:y_bd2,x_bd1:x_kt1-1] = 3 
KQ[:,y_bd2:y_bd3,x_bd2:x_kt2] = 3 
KQ[:,y_bd3:y_bd4,x_bd3:x_kt3] = 3
KQ[:,y_bd4:y_kt+1,x_bd4:x_kt4] = 3 


# ## xác định trạng thái của vật đúc dựa trên nhiệt độ

# In[14]:


def update_KQ4(t):
    condition1 = (VL[t, 0:-1,1 :-1] == 1) & (T[t, 0:-1, 1:-1] > 660) #dieu kien vat duc long
    condition2 = (VL[t, 0:-1, 1:-1] == 1) & ((T[t, 0:-1, 1:-1] <= 660) & (T[t, 0:-1, 1:-1] >= 580)) #dieu kien vat duc ban long
    condition3 = (VL[t, 0:-1, 1:-1] == 1) & (T[t, 0:-1, 1:-1] < 580) #dieu kien vat duc ran
    condition4 = (VL[t, 0:-1, 1:-1] == 0)                      #dieu kien khuon

    choices = [3, 2, 1, 0]  # 3-long,2-ban long, 1 -ran ,0- khuon

    KQ[t+1, 0:-1, 1:-1] = np.select([condition1, condition2, condition3, condition4], choices, KQ[t, 0:-1, 1:-1])


# ## cập nhật giá trị theo pha và tính tỷ phần pha - % đông đặc sau 1 bước

# In[15]:


def update_F3(t):
    condition_long = (KQ[t, 0:-1, 1:-1] == 3)
    condition_banlong = (KQ[t, 0:-1, 1:-1] == 2)
    condition_ran = (KQ[t, 0:-1, 1:-1] == 1)
    condition_khuon = (KQ[t, 0:-1, 1:-1] == 0)
    F[t+1,0:-1,1:-1] = np.select(
        [ condition_long,
          condition_banlong,
          condition_ran,   
          condition_khuon,
        ],
        [
            F2,        # condition_long
            F2pha,        # condition_banlong
            F3,        # condition_ran 
            F1         # condition_khuon
        ],
        F[t,0:-1,1:-1]
    )
    


# In[16]:


def tinh_toan_a(t):
    if KQ[t, 0:-1, 1:-1].any() == 2 :
        return (T[t-1, 0:-1, 1:-1] - T[t, 0:-1, 1:-1]) / 80
    else:
        return 0
a = tinh_toan_a(t) 


# In[17]:


def tinh_toan_b(t):
    if KQ[t, 0:-1, 1:-1].any() == 2 :
        return (660 - T[t, 0:-1, 1:-1]) / 80
    else:
        return 0
b = tinh_toan_b(t) 


# In[18]:


def tinhF2pha(t):
    if KQ[t, 0:-1, 1:-1].any() == 2:
        return (b*K3+(1-b)*K2)*dt/((b*p3+(1-b)*p2)*(b*Cp3+(1-b)*Cp2)*dx**2)
    else:
        return 0
F2pha = tinhF2pha(t)    


# In[19]:


def heso(t):
    if KQ[t, 0, 1:-1].any() == 2:
        return b*K3+(1-b)*K2
    elif KQ[t, 0:-1, 1:-1].any() == 3:
        return K2
    else: 
        return K3
K= heso(t)    


# ## các hàm tính tại góc, cạnh, lòng
# 

# In[20]:


def goc(t):
    T[t+1,0,0]= F1*(2*T[t,0,1]+2*T[t,1,0]-4*T[t,0,0]+4*Bio1*(T_env-T[t,0,0]))+T[t,0,0] #Góc dưới trái
    T[t+1,0,-1]= F1*(2*T[t,0,-2]+2*T[t,1,-1]-4*T[t,0,-1]+4*Bio1*(T_env-T[t,0,-1]))+T[t,0,-1] #Góc dưới phải
    T[t+1,-1,-1]= F1*(2*T[t,-2,-1]+2*T[t,-1,-2]-4*T[t,-1,-1]+4*Bio1*(T_env-T[t,-1,-1]))+T[t,-1,-1] #Góc trên phải
    T[t+1,-1,0]= F1*(2*T[t,-2,0]+2*T[t,-1,1]-4*T[t,-1,0]+4*Bio1*(T_env-T[t,-1,0]))+T[t,-1,0] #Góc trên trái   
    


# In[21]:


def canh3(t):
    A = T[t, 1 , 1:-1]  # duoi 
    D = T[t, 0 , :-2]    # trai
    C = T[t, 0, 2:]     # phai
    E = T[t, 0, 1:-1]

    # Chọn giá trị cho F tùy theo nhiệt độ
    F_d =  F[t, 1 , 1:-1]
    F_t =  F[t, 0, :-2]
    F_p =  F[t, 0, 2:]
    F_g =  F[t, 0, 1:-1]
    condition = (KQ[t, 0, 1:-1].any() == 0)
    # Chọn giá trị cho F tùy theo nhiệt độ
    result1 = F1*2*A + F1*D + F1*C -(2*F1  + F1 + F1)*E + a*(L/(b*Cp3+(1-b)*Cp2))+2*Bio1*F1*(T_env-E) + E 
    result2 = F_d*2*A + F_p*D + F_t*C -(2*F_d  + F_t + F_p)*E + a*(L/(b*Cp3+(1-b)*Cp2))+2*(h0*dx/K)*F_g*(T_env-E) + E 
    
    T[t+1, 0, 1:-1] = np.select(
        [ condition],
        [result1],
        result2
    )
    
   
    #Canh duoi
    A = T[t, -2 , 1:-1]
    D = T[t, -1  , :-2]
    C = T[t, -1  , 2:  ]
    E = T[t, -1  , 1:-1]
    T[t+1,-1,1:-1]= F1*(2*A+C+D-4*E+2*Bio1*(T_env-E))+E
    
    #Canh trái
    A = T[t, 1:-1 , 1]
    D = T[t,  :-2 , 0]
    C = T[t, 2:   , 0]
    E = T[t, 1:-1 , 0]
    T[t+1,1:-1,0]= F1*(2*A+C+D-4*E+2*Bio1*(T_env-E))+E

    #Canh phải
    A = T[t, 1:-1 , -2]
    D = T[t,  :-2 , -1]
    C = T[t, 2:   , -1]
    E = T[t, 1:-1 , -1]
    T[t+1,1:-1,-1]= F1*(2*A+C+D-4*E+2*Bio1*(T_env-E))+E


# In[22]:


def Long4(t):
    A = T[t, 2: , 1:-1]  # duoi 
    B = T[t, :-2, 1:-1]    # tren
    C = T[t, 1:-1, 2:]     # phai
    D = T[t, 1:-1, :-2]    # trai
    E = T[t, 1:-1, 1:-1]

    # Chọn giá trị cho F tùy theo nhiệt độ
    F_d =  F[t, 2:  , 1:-1]
    F_tr = F[t, :-2, 1:-1]
    F_t =  F[t, 1:-1, 2:]
    F_p =  F[t, 1:-1, :-2]

    # Chọn giá trị cho F tùy theo nhiệt độ
    T[t+1, 1:-1, 1:-1] = F_d*A + F_tr*B + F_p*D + F_t*C - (F_d + F_tr + F_t + F_p)*E + a*(L/(b*Cp3+(1-b)*Cp2)) + E 


# In[23]:


start = time.time()
#Tính lặp dùng vòng for
for t in range(0, step-1):
    update_KQ4(t)
    update_F3(t)
    tinh_toan_a(t)
    tinhF2pha(t)
    heso(t)
    goc(t)
    canh3(t)
    Long4(t)
print('Thoi gian tinh toan', time.time()-start, 's')


# # Kết quả

# ## Thời gian vật đúc đông đặc hết

# In[24]:


# Tìm thời điểm cuối cùng mà giá trị KQ bằng 2 xuất hiện
last_time_index = None
for t in range(step - 1, -1, -1):
    if 2 in KQ[t]:
        last_time_index = t
        break

# In ra kết quả
if last_time_index is not None:
    print("Thời điểm cuối cùng mà giá trị KQ bằng 2 xuất hiện là:", last_time_index)
else:
    print("Không có giá trị KQ bằng 2 trong mảng KQ.")


# ## Tọa độ điểm lõm co

# In[25]:


if last_time_index is not None:
    indices = np.where(KQ[last_time_index] == 2)
    if len(indices[0]) > 0:
        last_row_index = indices[0][-1]  # Lấy chỉ số cuối cùng trong list indices
        last_col_index = indices[1][-1]
        print("Tọa độ của điểm có giá trị KQ bằng 2 cuối cùng là:", (last_row_index, last_col_index))
    else:
        print("Không tìm thấy tọa độ của điểm có giá trị KQ bằng 2.")
else:
    print("Không có giá trị KQ bằng 2 trong mảng KQ.")


# ## biểu đồ nhiệt độ

# In[25]:


#Update du lieu
def update(time):
    temp.data_source.data['image'] = [np.flip(T[time],0)]
    push_notebook()
#     push_notebook(handle=t_blur)


# In[26]:


from bokeh.palettes import Turbo256
from bokeh.models import LinearColorMapper, ColorBar


#Tao khung hinh ve
plot = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
plot.x_range.range_padding = plot.y_range.range_padding = 0

#Tao bang mau
color_mapper = LinearColorMapper(palette=Turbo256, low=T_env, high=T1)
color_bar = ColorBar(color_mapper=color_mapper)

#Do thi nhiet do
temp = plot.image(image=[T[0]], x=0, y=0, dw=nx, dh=ny, color_mapper=color_mapper, level="image")
plot.grid.grid_line_width = 0.5 
show(plot, notebook_handle=True);

#Thiet lap thanh truot
time_slider = ipw.IntSlider(min=0,max=step-1, continuous_update=False)

#Hien thi
interact(update, time=time_slider);


# In[ ]:





# ## biểu đồ pha 

# In[26]:


def update1(time):
    subset_T = KQ[time, :,:]
    flipped_subset_T = np.flip(subset_T, 0)
    temp.data_source.data['image'] = [flipped_subset_T]
    # Push the update to the notebook
    push_notebook()


# In[27]:


from bokeh.palettes import Turbo256
from bokeh.models import LinearColorMapper, ColorBar

#Tao khung hinh ve
plot = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
plot.x_range.range_padding = plot.y_range.range_padding = 0

#Tao bang mau
color_mapping = {
    0: "black",
    1: "blue",
    2: "yellow",
    3:"red",
}

colors = list(color_mapping.values())  # Extract color values as a list
color_mapper = LinearColorMapper(palette=colors, low=0, high=3)

color_bar = ColorBar(color_mapper=color_mapper)

#Do thi nhiet do
temp = plot.image(image=[KQ[0]], x=0, y=0, dw=nx, dh=ny, color_mapper=color_mapper, level="image")
plot.grid.grid_line_width = 0.5 
show(plot, notebook_handle=True);
#Thiet lap thanh truot
time_slider = ipw.IntSlider(min=0,max=step-1, continuous_update=False)

#Hien thi
interact(update1, time=time_slider);


# ## đồ thị nhiệt độ tại các điểm tùy chỉnh

# In[29]:


import numpy as np
from bokeh.plotting import figure, show

# Lấy giá trị thời gian từ 0 đến step-1
time_values = np.arange(step)

# Lấy giá trị nhiệt độ tại điểm (, ) theo thời gian
temperature_values = T[:, 9, 26]

# Tạo figure
p = figure(title="Biểu đồ nhiệt độ tại các điểm của vật liệu đúc theo thời gian",
           x_axis_label='Thời gian',
           y_axis_label='Nhiệt độ')

# Vẽ đường cong biểu diễn nhiệt độ theo thời gian
p.line(time_values, temperature_values, line_width=2)

# Hiển thị đồ thị
show(p)


# In[42]:


import matplotlib.pyplot as plt

# Lấy giá trị thời gian từ 0 đến step-1
time_values = np.arange(step)

# Lấy giá trị nhiệt độ tại điểm (25, 25) theo thời gian
temperature_values = T[:, 0, 26]

# Tạo figure và axes
fig, ax = plt.subplots()

# Vẽ đường cong biểu diễn nhiệt độ theo thời gian
ax.plot(time_values, temperature_values, linewidth=2)

# Đặt nhãn cho trục x và y
ax.set_xlabel('Thời gian')
ax.set_ylabel('Nhiệt độ')

# Đặt tiêu đề cho biểu đồ
ax.set_title('Biến đổi nhiệt độ tại điểm (25, 25) theo thời gian')

# Hiển thị biểu đồ
plt.show()


# In[ ]:




