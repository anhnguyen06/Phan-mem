#!/usr/bin/env python
# coding: utf-8

# # MÔ PHỎNG QUÁ TRÌNH ĐÔNG ĐẶC CỦA VẬT ĐÚC

# ## KHAI BÁO THƯ VIỆN VÀ CÁC THÔNG SỐ ĐẦU VÀO

# In[33]:


#Các thư viện cần thiết
import numpy as np
import pandas as pd
import math as math
from ipywidgets import interact
import ipywidgets as ipw

import time


# In[34]:


## Nhập các giá trị từ bên ngoài
L_vatduc = float(input('Nhập chiều dài của vật đúc: '))
R_vatduc = float(input('Nhập bán kính của vật đúc: '))
L1 = float(input('Nhập chiều dài L1: '))
R1 = float(input('Nhập bán kính R1: '))
L2 = float(input('Nhập chiều dài L2: '))
R2 = float(input('Nhập bán kính R2: '))
L3 = float(input('Nhập chiều dài L3: '))
R3 = float(input('Nhập bán kính R3: '))
L4 = float(input('Nhập chiều dài L4: '))
R4 = float(input('Nhập bán kính R4: '))
L_khuon = float(input('Nhập chiều dài của khuôn: '))
R_khuon = float(input('Nhập bán kính của khuôn: '))

# In các giá trị đã nhập để kiểm tra
print('Các giá trị đã nhập:')
print('Chiều dài của vật đúc:', L_vatduc)
print('Bán kính của vật đúc:', R_vatduc)
print('L1:', L1)
print('R1:', R1)
print('L2:', L2)
print('R2:', R2)
print('L3:', L3)
print('R3:', R3)
print('L4:', L4)
print('R4:', R4)
print('Chiều dài của khuôn:', L_khuon)
print('Bán kính của khuôn:', R_khuon)


# In[35]:


# Nhập các giá trị từ bên ngoài
K1 = float(input("Nhập hệ số dẫn nhiệt cho thép: "))
Cp1 = float(input("Nhập nhiệt dung riêng cho thép: "))
p1 = float(input("Nhập khối lượng riêng cho thép: "))

K2 = float(input("Nhập hệ số dẫn nhiệt cho nhôm lỏng: "))
Cp2 = float(input("Nhập nhiệt dung riêng cho nhôm lỏng: "))
p2 = float(input("Nhập khối lượng riêng cho nhôm lỏng: "))

K3 = float(input("Nhập hệ số dẫn nhiệt cho nhôm đặc: "))
Cp3 = float(input("Nhập nhiệt dung riêng cho nhôm đặc: "))
p3 = float(input("Nhập khối lượng riêng cho nhôm đặc: "))

# In các giá trị đã nhập để kiểm tra
print("Các giá trị đã nhập:") 
print("Thép - K1:", K1) # [w/mk]  Hệ số dẫn nhiệt
print("Thép - Cp1:", Cp1) #[j/kg.k]  Nhiệt dung riêng
print("Thép - p1:", p1) #[Kg/m^3] Khối lượng riêng
print("Nhôm lỏng - K2:", K2) # [w/mk] Hệ số dẫn nhiệt
print("Nhôm lỏng - Cp2:", Cp2) #[j/kg.k] Nhiệt dung riêng
print("Nhôm lỏng - p2:", p2) # [Kg/m^3] Khối lượng riêng
print("Nhôm đặc - K3:", K3) # [w/mk] Hệ số dẫn nhiệt
print("Nhôm đặc - Cp3:", Cp3) #[j/kg.k]  Nhiệt dung riêng 
print("Nhôm đặc - p3:", p3) # [Kg/m^3] Khối lượng riêng


# In[36]:


# Nhập giá trị hệ số tỏa nhiệt ra môi trường
h0 = float(input("Nhập hệ số tỏa nhiệt ra môi trường: "))
h1 = float(input("Nhập hệ số tỏa nhiệt ra môi trường khác (nếu có): "))
L = float(input("Nhập giá trị L: "))

# In các giá trị đã nhập để kiểm tra
print("Các giá trị đã nhập:")
print("Hệ số tỏa nhiệt ra môi trường:", h0)
print("Hệ số tỏa nhiệt ra môi trường khác (nếu có):", h1)
print("Giá trị L:", L)


# In[37]:


# Hệ số dẫn nhiệt tại ranh giới khuôn và vật đúc 
K_bg2 = (K1+K2)/2 # lỏng - khuôn
K_bg3 = (K1+K3)/2 # rắn - khuôn 
print('K_bg2:', K_bg2)
print('K_bg3:', K_bg3)


# In[38]:


# Nhập các giá trị nhiệt độ ban đầu và môi trường
T1 = float(input("Nhập nhiệt độ ban đầu của vật đúc (T1): "))
T_kettinh = float(input("Nhập nhiệt độ kết tinh của vật đúc (T_kettinh): "))
T_khuon = float(input("Nhập nhiệt độ ban đầu của khuôn thép (T_khuôn): "))
T_env = float(input("Nhập nhiệt độ môi trường (T_env): "))

# In các giá trị đã nhập để kiểm tra
print("Các giá trị đã nhập:")
print("Nhiệt độ ban đầu của vật đúc (T1):", T1)
print("Nhiệt độ kết tinh của vật đúc (T_kettinh):", T_kettinh)
print("Nhiệt độ ban đầu của khuôn thép (T_khuôn):", T_khuon)
print("Nhiệt độ môi trường (T_env):", T_env)


# In[39]:


# Nhập các giá trị dx và dy
dx = float(input("Nhập giá trị của dx: "))
dy = float(input("Nhập giá trị của dy: "))

# In các giá trị đã nhập để kiểm tra
print("Các giá trị đã nhập:")
print("Giá trị của dx:", dx)
print("Giá trị của dy:", dy)


# In[40]:


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


# In[41]:


# Chọn lại dt <= dt_ondinh
dt=math.pow(10,math.floor(math.log10(dt_ondinh)))
dt


# In[42]:


# Tính Fourier 
F1 = (alpha1*dt)/dx**2
F2 = (alpha2*dt)/dx**2
F3 = (alpha3*dt)/dx**2
print(' Khuon :', F1)
print(' vat duc long :', F2)
print(' vat duc ran :', F3)


# In[43]:


#thoi gian tinh toan 
t = 50
step = int(t/dt) 
step


# ## TẠO MÔ HÌNH CHO VẬT ĐÚC VÀ KHUÔN ĐÚC

# In[44]:


T = np.zeros((int(step), int(ny)+1, int(nx)+1))+ T_env
x_bd1 = int((nx-nx1)/2 +0.5)
x_bd2 = int((nx-nx2)/2 +0.5)
x_bd3 = int((nx-nx3)/2 +0.5)
x_bd4 = int((nx-nx4)/2 +0.5)
x_kt1 = int(x_bd1+ nx1)
x_kt2 = int(x_bd2+ nx2)
x_kt3 = int(x_bd3+ nx3)
x_kt4 = int(x_bd4+ nx4)
y_bd1= int((ny-ny_vatduc)/2 +0.5)
y_bd2 = y_bd1+ ny1
y_bd3= y_bd2 + ny2
y_bd4 = y_bd3 + ny3
y_kt = int(y_bd1+ ny_vatduc)
print(x_bd1,x_bd2,x_bd3,x_bd4,x_kt1,x_kt2,x_kt3,x_kt4)
print(y_bd1,y_bd2,y_bd3,y_bd4,y_kt)
print(T.shape)  


# In[45]:


T[:,y_bd1:y_bd2,x_bd1:x_kt1+1] = T1 
T[:,y_bd2:y_bd3,x_bd2:x_kt2+1] = T1 
T[:,y_bd3:y_bd4,x_bd3:x_kt3+1] = T1
T[:,y_bd4:y_kt+1,x_bd4:x_kt4+1] = T1 


# In[46]:


VL = np.zeros((int(step), int(ny)+1, int(nx)+1)) #vì chạy từ 0 nên chỉ cần +1 nút giao biên và 2 gia biên cho y,x
VL[:,y_bd1:y_bd2,x_bd1:x_kt1+1] = 1 
VL[:,y_bd2:y_bd3,x_bd2:x_kt2+1] = 1 
VL[:,y_bd3:y_bd4,x_bd3:x_kt3+1] = 1
VL[:,y_bd4:y_kt+1,x_bd4:x_kt4+1] = 1 


# In[47]:


F = np.zeros((int(step), int(ny)+1, int(nx)+1)) +F1 
F[:,y_bd1:y_bd2,x_bd1:x_kt1+1] = F2 
F[:,y_bd2:y_bd3,x_bd2:x_kt2+1] = F2 
F[:,y_bd3:y_bd4,x_bd3:x_kt3+1] = F2
F[:,y_bd4:y_kt+1,x_bd4:x_kt4+1] = F2 


# In[48]:


KQ = np.zeros((int(step), int(ny)+1, int(nx)+1))
KQ[:,y_bd1:y_bd2,x_bd1:x_kt1+1] = 3 
KQ[:,y_bd2:y_bd3,x_bd2:x_kt2+1] = 3 
KQ[:,y_bd3:y_bd4,x_bd3:x_kt3+1] = 3
KQ[:,y_bd4:y_kt+1,x_bd4:x_kt4+1] = 3 


# ### XÉT TRẠNG THÁI CỦA VẬT ĐÚC

# In[49]:


def update_KQ4(t):
    condition1 = (VL[t, 1:-1,1 :-1] == 1) & (T[t, 1:-1, 1:-1] > 660) #dieu kien vat duc long
    condition2 = (VL[t, 1:-1, 1:-1] == 1) & ((T[t, 1:-1, 1:-1] <= 660) & (T[t, 1:-1, 1:-1] >= 580)) #dieu kien vat duc ban long
    condition3 = (VL[t, 1:-1, 1:-1] == 1) & (T[t, 1:-1, 1:-1] < 580) #dieu kien vat duc ran
    condition4 = (VL[t, 1:-1, 1:-1] == 0)                      #dieu kien khuon

    choices = [3, 2, 1, 0]  # 3-long,2-ban long, 1 -ran ,0- khuon

    KQ[t+1, 1:-1, 1:-1] = np.select([condition1, condition2, condition3, condition4], choices, KQ[t, 1:-1, 1:-1])


# ## CẬP NHẬT CÁC GIÁ TRỊ F

# In[50]:


def update_F3(t):
    condition_long = (KQ[t, 1:-1, 1:-1] == 3)
    condition_banlong = (KQ[t, 1:-1, 1:-1] == 2)
    condition_ran = (KQ[t, 1:-1, 1:-1] == 1)
    condition_khuon = (KQ[t, 1:-1, 1:-1] == 0)
    F[t+1,1:-1,1:-1] = np.select(
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
        F[t,1:-1,1:-1]
    )
    


# ## TÍNH TOÁN GIÁ TRỊ % PHA LỎNG TẠI VÙNG BÁN LỎNG VÀ F TẠI VÙNG BÁN LỎNG

# In[51]:


def tinh_toan_a(t):
    if KQ[t, 1:-1, 1:-1].any() == 2 :
        return (T[t-1, 1:-1, 1:-1] - T[t, 1:-1, 1:-1]) / 80
    else:
        return 0
a = tinh_toan_a(t) 


# In[52]:


def tinh_toan_b(t):
    if KQ[t, 1:-1, 1:-1].any() == 2 :
        return (660 - T[t, 1:-1, 1:-1]) / 80
    else:
        return 0
b = tinh_toan_b(t) 


# In[53]:


def tinhF2pha(t):
    if KQ[t, 1:-1, 1:-1].any() == 2:
        return (b*K3+(1-b)*K2)*dt/((b*p3+(1-b)*p2)*(b*Cp3+(1-b)*Cp2)*dx**2)
    else:
        return 0
F2pha = tinhF2pha(t)    


# ## TÍNH TOÁN NHIỆT ĐỘ 

# In[54]:


def goc(t):
    T[t+1,0,0]= F1*(2*T[t,0,1]+2*T[t,1,0]-4*T[t,0,0]+4*Bio1*(T_env-T[t,0,0]))+T[t,0,0] #Góc dưới trái
    T[t+1,0,-1]= F1*(2*T[t,0,-2]+2*T[t,1,-1]-4*T[t,0,-1]+4*Bio1*(T_env-T[t,0,-1]))+T[t,0,-1] #Góc dưới phải
    T[t+1,-1,-1]= F1*(2*T[t,-2,-1]+2*T[t,-1,-2]-4*T[t,-1,-1]+4*Bio1*(T_env-T[t,-1,-1]))+T[t,-1,-1] #Góc trên phải
    T[t+1,-1,0]= F1*(2*T[t,-2,0]+2*T[t,-1,1]-4*T[t,-1,0]+4*Bio1*(T_env-T[t,-1,0]))+T[t,-1,0] #Góc trên trái   
    


# In[55]:


def canh(t):
    A = T[t, 1 , 1:-1] #duoi
    D = T[t, 0  , :-2] #trai
    C = T[t, 0  , 2:  ] #phai 
    E = T[t, 0  , 1:-1] #giua
    T[t+1,-1,1:-1]= F1*(2*A+C+D-4*E+2*Bio1*(T_env-E))+E
   
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


# In[56]:


def canh2(t):
    A = T[t, 1 , 1:-1] #duoi
    D = T[t, 0  , :-2] #trai
    C = T[t, 0  , 2:  ] #phai 
    E = T[t, 0  , 1:-1] #giua
    T[t+1,0,1:-1]= F1*(2*A+C+D-4*E+2*Bio1*(T_env-E))+E
   
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


# In[57]:


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
    T[t+1, 1:-1, 1:-1] = F_d*A + F_tr*B + F_p*D + F_t*C - (F_d + F_tr + F_t + F_p)*E + a*(L/(Cp3)) + E 


# In[58]:


start = time.time()
#Tính lặp dùng vòng for
for t in range(0, step-1):
    update_KQ4(t)
    update_F3(t)
    tinh_toan_a(t)
    tinhF2pha(t)
    goc(t)
    canh2(t)
    Long4(t)
print('Thoi gian tinh toan', time.time()-start, 's')


# In[59]:


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


# In[60]:


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


# In[ ]:




