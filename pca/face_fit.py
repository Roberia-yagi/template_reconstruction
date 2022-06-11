from geo import *

import numpy
import numpy.linalg

def calc_f_diff(trans,fp_data) :
  n_pos = 0
  for ff in fp_data :
    n_pos=n_pos+len(ff[1])
  f_diff = numpy.empty((n_pos),dtype=numpy.float64)
  inv_o_trans = -trans
  idx=0
  for ff in fp_data :
    inv_s_trans= -ff[0][1]
    for pp in ff[1] :
      tmp=inv_o_trans*pp
      tmp=inv_s_trans*tmp
      if ff[0][0] == "plane" :
        f_diff[idx]=tmp[2]
      elif ff[0][0] == "cylinder" :
        f_diff[idx]=sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])-ff[0][2]
      elif ff[0][0] == "sphere" :
        f_diff[idx]=abs(tmp)-ff[0][2]
      idx += 1
  return f_diff

def calc_common_jacob(inv_o, inv_s, pp) :
  result = []
  result.append(-(inv_s.mat.col(0)))
  result.append(-(inv_s.mat.col(1)))
  result.append(-(inv_s.mat.col(2)))
  tmp=inv_o*pp
  result.append(-(inv_s*VECTOR(0,-tmp[2],tmp[1])))
  result.append(-(inv_s*VECTOR(tmp[2],0,-tmp[0])))
  result.append(-(inv_s*VECTOR(-tmp[1],tmp[0],0)))
  return result

def calc_jacob(trans, fp_data) :
  n_pos = 0
  for ff in fp_data :
    n_pos=n_pos+len(ff[1])
  jacob = numpy.empty((n_pos,6),dtype=numpy.float64)
  inv_o_trans = -trans
  idx=0
  for ff in fp_data :
    inv_s_trans= - ff[0][1]
    for pp in ff[1] :
      tmp=inv_o_trans*pp
      tmp=inv_s_trans*tmp
      c_jacob=calc_common_jacob(inv_o_trans,inv_s_trans,pp)     
      if ff[0][0] == "plane" :
        tmp = VECTOR(0,0,1)
      elif ff[0][0]== "cylinder" :
        tmp = VECTOR(tmp[0]/sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]),
                     tmp[1]/sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]),
                     0)
      elif ff[0][0] == "sphere" :
        tmp = 1.0/abs(tmp)*tmp      
      for i in range(6) :
        jacob[idx][i]= tmp.dot(c_jacob[i])
      idx += 1
  return jacob

def fit_face1(trans0, fp_data, rlim=0.001) :
  f_diff=calc_f_diff(trans0, fp_data)
  jacob = calc_jacob(trans0, fp_data)
  u_mat, w_vec, v_trn=numpy.linalg.svd(jacob,compute_uv=1,full_matrices=0)
  w_max=max(w_vec)
  w_min=w_max*rlim
  w_inv=numpy.empty((6),dtype=numpy.float64)
  for j in range(6) :
    if w_vec[j]<w_min :
      w_inv[j]=0.0
    else :
      w_inv[j]=1.0/w_vec[j]
  u_trn=u_mat.transpose()
  tmp=numpy.dot(u_trn,f_diff)
  tmp=tmp*w_inv
  v_mat=v_trn.transpose()
  q=numpy.dot(v_mat,tmp)
  q_xyzabc=[-q[0],-q[1],-q[2],-q[3],-q[4],-q[5]]
  d_trans=FRAME(xyzabc=q_xyzabc)
  trans1=trans0*d_trans
  return trans1,q_xyzabc  
  
