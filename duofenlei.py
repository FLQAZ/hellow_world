# 手写数字问题
import  numpy as np
import matplotlib.pyplot as plt
import  scipy.io as sio

data=sio.loadmat('ex3data1.mat')  # data是一个字典的格式
print(data)
# print(type(data))
# print(data.keys())

raw_X=data['X']
raw_y=data['y']
# print(raw_X)
# print(raw_y)
# print(raw_X.shape,raw_y.shape)
# 得出的结果是(5000，400),(5000，1)前者表示的是
# 有五千个样本和400个特征,400个特征表示的是20*20的图片像素点


def plot_an_image(X,y):
    pick_one=np.random.randint(5000)
    image=X[pick_one,:]
    fig,ax=plt.subplots(figsize=(1,1))
    ax.imshow(image.reshape(20,20).T,cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(y[pick_one]))
# plot_an_image(raw_X,raw_y)



def  plot_100_image(X):
    sample_index=np.random.choice(len(X),100)
    images =X[sample_index,:]
    fig,ax=plt.subplots(ncols=10,nrows=10,figsize=(8,8),sharex=True,sharey=True)

    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(images[10*i+j].reshape(20,20).T,
                           cmap='gray_r')

    plt.xticks([])
    plt.yticks([])
    plt.show()

# plot_100_image(raw_X)


def  sigmoid(z):
    return  1/(1+np.exp(-z))

def  costFuction(theta,X,y,lamda):
    A=sigmoid(X@theta)

    first=y*np.log(A)            # 这里的*表示的是对应元素相乘
    # 对于多分类问题，而言，y=1,2,3,4,5 不同的标签，如果A为1的话，
    # 则log(A)为0，这样y*log(A)为0，这样损失函数值最小，跟y的值无光
    second=(1-y)*np.log(1-A)
    reg=theta[1:]@theta[1:]*(lamda/(2*len(X)))

    return  -np.sum(first+second)/len(X)+reg



def gradient_reg(theta,X,y,lamda):
    reg=theta[1:]*(lamda/len(X))
    reg=np.insert(reg,0,values=0,axis=0)

    first=(X.T@(sigmoid(X@theta)-y))/len(X)

    return  first+reg

X=np.insert(raw_X,0,values=1,axis=1)
print(X.shape)
y=raw_y.flatten()
print(y.shape)


from scipy.optimize import  minimize

def one_vs_all(X,y,lamda,K):
    n=X.shape[1]

    theta_all=np.zeros((K,n))

    for i in range(1,K+1):
        theta_i =np.zeros(n,)
        res=minimize(fun=costFuction,
                     x0=theta_i,
                     args=(X,y==i,lamda),
                     method='TNC',
                     jac=gradient_reg)
        theta_all[i-1,:]=res.x
    return  theta_all

lamda =1
K=10

theta_final=one_vs_all(X,y,lamda,K)
# print(theta_final)
print(theta_final.shape)
print(X@theta_final.T)
'''
def predict(X,theta_final):
    h=sigmoid(X@theta_final.T)  #x(5000,401) theta_final(10,401)
    h_argmax=np.argmax(h,axis=1)   
    # 这个函数在这里表示的是h(5000,10),每一行的所有列比较，
    # 返回h最大值的编号，如3号的概率最大，返回3
    return  h_argmax+1

y_pred=predict(X,theta_final)
acc=np.mean(y_pred==y)
acc
print(acc)
'''