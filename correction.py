# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:40:57 2018

@author: wpc
"""
# from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


#
##-------------------------Hough transformation----------------------------
def correction_model(imgs):
    img_gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    
    img_gray = cv2.GaussianBlur(img_gray,(3,3),0) 
    
    img_canny = cv2.Canny(img_gray, 100 , 200)
    
    l,m=img_canny.shape
#-----------HoughLinesP---------------------    
    lines2 = cv2.HoughLinesP(img_canny,1,np.pi/180,30,minLineLength=100,maxLineGap=10) # 参数：图像、rho的步长、theta的步长、有效线判断的最小点数、最小直线数、线段合并的最大间距
    
    #--------------delete adjacent lines删除相邻的线段--------------------
    fp1=[]
    fp2=[]
    lines3=[]
    for i in range(int(lines2.shape[0])):
        x1,y1,x2,y2 = lines2[i][0]
        fp1+=[(x1,y1,x2,y2)]       #-------------delete adjacent lines on vertical direction-------------
    
        fp2+=[(y1,x1,x2,y2)]        #-------------delete adjacent lines on horizontal direction-------------
    
    fp1.sort()
    fp2.sort()   
    ind=0
    for i in range(len(fp1)-1):
        if abs(fp1[ind+1][0]-fp1[ind][0])<10:           #-----------if the spacing  of two adjacent lines is less than 10, then delete两条线的间距如果小于10，则删除
            del fp1[ind]
            ind-=1
        ind+=1
    
    ind=0
    for i in range(len(fp2)-1):
        if abs(fp2[ind+1][0]-fp2[ind][0])<10:
            del fp2[ind]
            ind-=1
        ind+=1    
    
    for y1,x1,x2,y2 in fp2:                 #---------put the flitered lines together将所有符合条件的线段合并到一起-----------
        fp1+=[(x1,y1,x2,y2)]
        
    
    
    # for x1,y1,x2,y2 in fp1:         #----------画出所有符合条件的线段
        
    #     cv2.line(imgs,(x1,y1),(x2,y2),(255,255,0),2)
    
    
    #------------calculate the expression of lines: y=ax+b------------
    slope=[]
    intercept=[]
    for x1,y1,x2,y2 in fp1:
        if x2!=x1:
            slope1=(y2-y1)/(x2-x1)
            intercept1=y1-x1*slope1
            slope+=[slope1]
            intercept+=[intercept1]
    
    #---------------divided these lines into two direction------------
    hor_line=[]
    ver_line=[]
    for ind in range(len(slope)):
        if slope[ind]<0.5 and slope[ind]>-0.5:
            hor_line+=[(slope[ind],intercept[ind])]
        elif slope[ind]>np.tan(np.pi*1/3) or slope[ind]<np.tan(np.pi*2/3):
            ver_line+=[(slope[ind],intercept[ind])]
    
    #---------------------calculate vashing points in horizontal direction------------------
    vanish_point2=[]
    i=0
    for slope1,intercept1 in hor_line[0:-1:10]:
        i+=1
        for slope2,intercept2 in hor_line[i::10]:
            if slope1-slope2!=0:
                xv=(intercept2-intercept1)/(slope1-slope2)          #两条直线的交点为灭点
                yv=slope1*xv+intercept1
                if xv!=np.inf and xv!=-np.inf:
                    vanish_point2+=[(xv,yv)]         #计算出来的灭点集合
    
    #---------------calculate vashing points in horizontal direction最小二乘法计算水平方向上所有灭点集合的平均值从而确定灭点位置----------------------
    sumx=0
    sumy=0                
    plt.figure(figsize=(8,6))
    for Xi,Yi in vanish_point2:
        sumx+=Xi
        sumy+=Yi
        plt.scatter(Xi,Yi,color="green",label="asd",linewidth=1)
    xch=sumx/len(vanish_point2)
    ych=sumy/len(vanish_point2)
    #print 'xc1=',xch,   'yc1=',ych
    
    def f_circle(vanish_point1,xc,yc):
        i=-1
        fall=[]
        for  x,y in vanish_point1:
            i+=1
            f=(x-xc)*(x-xc)+(y-yc)*(y-yc)-90000
            fall+=[(f,i)]
        fall.sort()
        ind=fall[-1][1]
        if fall[-1][0]>0:
            del vanish_point1[ind]
        return vanish_point1,i
    
    def center_circle(vanish_point1):
        sumx=0
        sumy=0
        for Xi,Yi in vanish_point1:
            sumx+=Xi
            sumy+=Yi
        xc=sumx/len(vanish_point1)
        yc=sumy/len(vanish_point1)
        return xc,yc
    
    for i in range(20):
        xc2,yc2=center_circle(vanish_point2)
        vanish_point2,ind=f_circle(vanish_point2,xc2,yc2)
        print (xc2,yc2)
            
    #    if ind==len(vanish_point2)-1:
    #        break
    
    
    plt.scatter(xch,ych,color="yellow",label="asd",linewidth=2)
    plt.scatter(xc2,yc2,color="blue",label="asd",linewidth=2)
    # plt.hold
    theta = np.arange(0, 2 * np.pi + 0.1,2 * np.pi / 1000)
    x=300*np.cos(theta)+xc2
    y=300*np.sin(theta)+yc2
    plt.plot(x, y, color='red')
    plt.show()
    
    #------------------connect each points on the left boundaty with horizontal vanishing point从水平灭点出发在图像上画出到图像右边缘的所有直线------------------
    connect_line1=[]
    if xc2<=0:
        for i in range(0,int(l),1):
            a1=(i-yc2)/(m-xc2)
            b1=yc2-a1*xc2
            connect_line1+=[(a1,b1)]
    #        cv2.line(imgs,(int(xc2),int(yc2)),(int(m),i),(255,0,0),2)
    else:
        for i in range(0,int(l),1):
            a1=(i-yc2)/(0-xc2)
            b1=yc2-a1*xc2
            connect_line1+=[(a1,b1)]
    #        cv2.line(imgs,(int(xc2),int(yc2)),(0,i),(255,0,0),2)
    
    
    #---------------------calculate vashing points in vertical direction计算垂直方向上的所有线段相交的灭点集合------------------
    vanish_point1=[]
    i=0
    for slope1,intercept1 in ver_line[0:-1:10]:
        i+=1
        for slope2,intercept2 in ver_line[i::10]:
            if slope1-slope2!=0:
                xv=(intercept2-intercept1)/(slope1-slope2)          #两条直线的交点为灭点
                yv=slope1*xv+intercept1
                if xv!=np.inf and xv!=-np.inf:
                    vanish_point1+=[(xv,yv)]         #计算出来的灭点集合
    
    #---------------calculate vashing points in horizontal direction利用最小二乘法计算垂直方向上所有灭点集合的平均值从而确定灭点位置----------------------
    sumx=0
    sumy=0
    plt.figure(figsize=(8,6))
    for Xi,Yi in vanish_point1:
        sumx+=Xi
        sumy+=Yi
        plt.scatter(Xi,Yi,color="green",label="asd",linewidth=1)
    xcv=sumx/len(vanish_point1)
    ycv=sumy/len(vanish_point1)
    
    
    for i in range(20):
        xc,yc=center_circle(vanish_point1)
        vanish_point1,ind=f_circle(vanish_point1,xc,yc)
    #    if ind==len(vanish_point1)-1:
    #        break
    
    
    plt.scatter(xcv,ycv,color="yellow",label="asd",linewidth=2)
    plt.scatter(xc,yc,color="red",label="asd",linewidth=2)
    # plt.hold
    theta = np.arange(0, 2 * np.pi + 0.1,2 * np.pi / 1000)
    x=300*np.cos(theta)+xc
    y=300*np.sin(theta)+yc
    plt.plot(x, y, color='red')
    plt.show()
    
    
    #------calculate the position of camera in oxy coordinate system坐标平面的位置-----------
    
    if xc2<=0:
        x2=int(m)
    else:
        x2=0
    
    y2=0
    
    sloped=1.2#-connect_line1[0][0]
    thetad=-np.arctan(sloped)
    b1d=sloped*x2
    L=int(m)
    if xc2<=0:
        x3o2=0
        y3o2=0
        x2o2=int(m)
        y2o2=0
        x3=x2-L/np.sqrt(1+sloped*sloped)
        y3=-sloped*x3+b1d
    else:
        x3o2=int(m)
        y3o2=0
        x2o2=0
        y2o2=0
        x3=x2+L/np.sqrt(1+sloped*sloped)
        y3=-sloped*x3+b1d
    
    #set the fielf of view as 45设视场角为45度
    thetad1=np.pi*3/8
    
    #已知等腰三角形两底角坐标和夹角，顶点计算公式计算xs,ys
    #if xc2<=0:
    #    xs=np.abs(x2-x3)/2+np.cos(thetad)*L*np.tan(thetad1)/2
    #    ys=np.abs(y2-y3)/2+np.sin(thetad)*L*np.tan(thetad1)/2
    #else:
    #    xs=np.abs(x2-x3)/2-np.cos(thetad)*L*np.tan(thetad1)/2
    #    ys=np.abs(y2-y3)/2-np.sin(thetad)*L*np.tan(thetad1)/2
        
    xs=((x2o2+x3o2)/2)*np.cos(thetad)-((x2o2-x3o2)/2)*np.tan(thetad1)*np.sin(thetad)+x3
    ys=((x2o2+x3o2)/2)*np.sin(thetad)+((x2o2-x3o2)/2)*np.tan(thetad1)*np.cos(thetad)+y3
    plt.figure(figsize=(8,8))
    
    plt.scatter(xs,ys,color="blue",label="asd",linewidth=2)
    plt.scatter(x3,y3,color="black",label="asd",linewidth=2)
    plt.scatter(x2,y2,color="yellow",label="asd",linewidth=2)
    # plt.hold
    plt.plot((x3,xs), (y3,ys), color='red')
    plt.plot((x2,xs), (y2,ys), color='red')
    plt.plot((x3,x2), (y3,y2), color='red')
    plt.show()
    
    #-----------Calculate the new plane of photo correction in the horizontal direction计算水平方向上像片纠正的改正量
    slopeds=(ys-y3)/(xs-x3)
    bds=ys-slopeds*xs
    
    y1=0
    x1=(y1-bds)/slopeds
    
    linean=[]
    bn=[]
    
    if x1<x2: 
        for i in range(int(x1),int(x2),1):
            slopea=(ys-0)/(xs-i)
            ba=ys-slopea*xs
            linean+=[(slopea,ba)]
    else:
        for i in range(int(x2),int(x1),1):
            slopea=(ys-0)/(xs-i)
            ba=ys-slopea*xs
            linean+=[(slopea,ba)]
    
    a2=(y2-y3)/(x2-x3)
    b2=y2-a2*x2
    
    xn=[]
    yn=[]
    in_point=[]
    for a,b in linean:
        x=(b2-b)/(a-a2)
        y=a*x+b
        xn+=[x]
        x_im=x*np.cos(thetad)+y*np.sin(thetad)+x3
        y_im=y*np.cos(thetad)-x*np.sin(thetad)-y3
        in_point+=[(x_im,y_im)]
    
    #------------------connect each points on the bottom boundary with vertical vanishing point从竖直灭点出发在图像上画出到图像下边缘一条边缘的所有直线------------------
    m2=np.abs(x2-x1)
    connect_line2=[]
    for i in range(0,int(m2),1):
        a2=(0-yc)/(in_point[i][0]-xc)
        b2=yc-a2*xc
        connect_line2+=[(a2,b2)]
    #    cv2.line(imgs,(int(xc),int(yc)),(int(in_point[i][0]),int(l)),(255,0,255),2)    
    
    #------calculate the position of camera in oyz coordinate system计算摄影中心在oyz坐标平面的位置-----------
    x22=0
    y22=0#int(l)
    
    sloped2=connect_line2[-1][0]
    thetad2=np.arctan(sloped2)
    
    bld2=sloped2*x22
    L=int(l)
    x32=np.abs(x22-L/np.sqrt(1+sloped2*sloped2))
    y32=sloped2*x32+bld2
    x32o2=int(l)
    y32o2=0
    
    
    #设视场角为45度,则底角为np.pi*3/8
    thetad1=np.pi*2/8
    
    
    
    #已知等腰三角形两底角坐标和夹角，顶点计算公式计算xs,yx
    #xs2=(x22-x32)/2+np.cos(thetad2)*L*np.tan(thetad1)/2
    #ys2=(y22-y32)/2+np.sin(thetad2)*L*np.tan(thetad1)/2
    
    xs2=((x22+x32o2)/2)*np.cos(thetad2)-((x22-x32o2)/2)*np.tan(thetad1)*np.sin(thetad2)
    ys2=((x22+x32o2)/2)*np.sin(thetad2)+((x22-x32o2)/2)*np.tan(thetad1)*np.cos(thetad2)
    
    
    plt.figure(figsize=(8,8))
    
    plt.scatter(xs2,ys2,color="blue",label="asd",linewidth=2)
    plt.scatter(x32,y32,color="black",label="asd",linewidth=2)
    plt.scatter(x22,y22,color="yellow",label="asd",linewidth=2)
    # plt.hold
    plt.plot((x32,xs2), (y32,ys2), color='red')
    plt.plot((x22,xs2), (y22,ys2), color='red')
    plt.plot((x32,x22), (y32,y22), color='red')
    plt.show()
    
     
    #---------------Calculate the new plane of photo correction in the vertical direction
    
    slopeds2=(ys2-y32)/(xs2-x32)
    bds2=ys2-slopeds2*xs2
    
    x12=0
    if bds2>0:
        y12=-bds2
    else:
        y12=bds2
    
    linean2=[]
    bn=[]
    for i in range(int(y12),int(y22),1):
        slopea2=(ys2+i)/(xs2-0)
        ba2=ys2-slopea2*xs2
        linean2+=[(slopea2,ba2)]
    
    a22=(y22-y32)/(x22-x32)
    b22=y22-a22*x22
    
    xn2=[]
    yn2=[]
    in_point2=[]
    in_pointcrop=[]
    for a,b in linean2:
        x=(b22-b)/(a-a22)
        y=a*x+b
        xn2+=[x]
        xi=0
        yi=l-np.sqrt((x-x22)*(x-x22)+(y-y22)*(y-y22))
        in_point2+=[(xi,yi)]
        if y>0:
            in_pointcrop+=[(x,y)]  #裁剪图像----------------------
    lcrop=len(in_pointcrop)
    
    #Fix all straight lines drawn on the image from the horizontal vanishing point to the right edge of the image------------------
    connect_line12=[]
    l21=lcrop
    l2=np.abs(y22-y12)
    if xc2<=0:
        for i in range(int(0),int(l2),1):
            a1=(in_point2[i][1]-yc2)/(0-xc2)
            b1=yc2-a1*xc2
            connect_line12+=[(a1,b1)]
    #        cv2.line(imgs,(int(xc2),int(yc2)),(int(m),int(in_point2[i][1])),(255,0,0),2)
    else:
        for i in range(int(0),int(l2),1):
            a1=(in_point2[i][1]-yc2)/(0-xc2)
            b1=yc2-a1*xc2
            connect_line12+=[(a1,b1)]    
    #        cv2.line(imgs,(int(xc2),int(yc2)),(0,int(in_point2[i][1])),(255,0,0),2)
    
    
    #------------Calculate the intersection of all straight lines。--------------
    img_copy2=np.uint8(np.zeros((int(l2),int(m2),3)))
     
    intersecter_points=[]
    for a1,b1 in connect_line2:
        for a2,b2 in connect_line12:
            x=(b2-b1)/(a1-a2)
            y=a1*x+b1
            intersecter_points+=[(x,y)]
    
    ind=0
    for i in range(int(m2)):
        for j in range(int(l2)):
            if 0<int(intersecter_points[ind][1])<int(l) and 0<int(intersecter_points[ind][0])<int(m):
                img_copy2[j,i]=imgs[int(intersecter_points[ind][1]),int(intersecter_points[ind][0])]
            ind+=1
    return img_copys

#-------------显示图像---------------------    
imgs = cv2.imread('image/46.jpg')

#cv2.imshow('a',img_canny)
#cv2.namedWindow('bb',cv2.WINDOW_NORMAL)
#cv2.imshow('bb',imgs)
#
#cv2.namedWindow('aa',cv2.WINDOW_NORMAL)
#cv2.imshow('aa',img_copy2)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img_copys=correction_model(imgs)
cv2.imwrite('imgs.jpg',imgs)
cv2.imwrite('img_copy2.jpg',img_copys)