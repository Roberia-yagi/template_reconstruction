import random
random.seed(0)
from geo import *
from face_fit import *


import numpy
import numpy.linalg

def distort(pos,sigma) :
    return map(lambda x: x+random.normalvariate(0,sigma), pos)

def make_cylinder_point_cloud(n,r,l_range,angle_range,sigma=None,transform=None) :
    rslt = []
    for i in range(n) :
        zz=random.uniform(l_range[0],l_range[1])
        th=random.uniform(angle_range[0],angle_range[1])
        if sigma :
            tmp=VECTOR(vec=list(distort([r*cos(th),r*sin(th),zz],sigma)))
        else :
            tmp=VECTOR(r*cos(th),r*sin(th),zz)
        if transform :
            tmp=transform*tmp
        rslt.append(tmp)
    return rslt

def calc_pca(point_cloud) :
    tmp=numpy.array(point_cloud).T
    tmp=numpy.cov(tmp,bias=True)
    return numpy.linalg.eig(tmp)

def calc_axis_center_pca(point_cloud) :
    tmp=calc_pca(point_cloud)
    idx=numpy.argmax(tmp[0])
    cc=numpy.average(point_cloud,axis=0)
    return VECTOR(tmp[1][0][idx], tmp[1][1][idx],
                  tmp[1][2][idx]),VECTOR(cc[0],cc[1],cc[2])

def make_cylinder_face_data(point_cloud,radius,transform=None) :
    if not transform :
        transform=FRAME()
    return [["cylinder",transform,radius],point_cloud]

def outlier(p_list, sigma, n=1) :
    rslt=[]
    for pp in p_list :
        for i in range(n) :
            rslt.append( VECTOR(pp[0]+random.normalvariate(0,sigma),
                                pp[1]+random.normalvariate(0,sigma),
                                pp[2]+random.normalvariate(0,sigma)))
    return rslt

def cmp_with_tv(tv, axis, center) :
    tv_axis=tv.mat.col(2)
    tmp1=abs(axis*tv_axis)
    tmp2=(-tv)*center
    return tmp1, tmp2
