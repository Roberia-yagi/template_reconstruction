from math import *

def sign(x) :
   if x > 0 :
      return 1
   elif x < 0 :
      return -1
   else :
      return 0

class VECTOR(list) :
  def dot(self, other) : # inner product
    return(self[0]*other[0] + 
           self[1]*other[1] +
           self[2]*other[2])
  def __neg__(self) : # negative vector
    return(VECTOR(-self[0],-self[1],-self[2]))
  def __add__(self, other) : # addition
    return VECTOR(self[0]+other[0],self[1]+other[1],self[2]+other[2])
  def __radd__(self, other) : # addtion to avoid list concatenation
    return VECTOR(self[0]+other[0],self[1]+other[1],self[2]+other[2])     
  def __sub__(self, other) : # subtraction
    return VECTOR(self[0]-other[0],self[1]-other[1],self[2]-other[2])
  def __mul__(self, other) : # cross protuct
    if not isinstance(other, VECTOR) :
       raise TypeError('error')
    tmp = VECTOR();
    tmp[0] = self[1]*other[2] - self[2]*other[1]
    tmp[1] = self[2]*other[0] - self[0]*other[2]
    tmp[2] = self[0]*other[1] - self[1]*other[0]
    return(tmp)
  def __rmul__(self, other) : # multiplication of scalar and MATRIX(this is defined in MATRIX)
    tmp = VECTOR();
    if isinstance(other, VECTOR) :
      pass
    elif isinstance(other, MATRIX) :
      pass
    else :
      for i in [0,1,2] :
        tmp[i] = other * self[i]
    return(tmp)
  def __abs__(self) : # absolute value
    return(sqrt(self.dot(self)))
  def normalize(self) : # normalization
    return 1.0/abs(self)*self
  def __repr__(self) : 
    return("v:" + list.__repr__(self))
  def __init__(self, x=0.0, y=0.0, z=0.0, vec=[]) :
    # vec is provided  to create a copy of a VECTOR.
    list.__init__(self,[float(x),float(y),float(z)])
    if vec:
      for i in [0,1,2] :
        self[i] = vec[i]

class MATRIX(list) :
  def col(self, idx, arg=[]) : # extraction of column VECTOR
    tmp = VECTOR()
    if arg :
      for i in [0,1,2] :
        self[i][idx] = arg[i]
    for i in [0,1,2] :
      tmp[i] = self[i][idx]
    return(tmp)
  def row(self, idx, arg=[]) : # extraction of row VECTOR
    tmp = VECTOR()
    if arg :
      for i in [0,1,2] :
        self[idx][i] = arg[i]
    for i in [0,1,2] :
      tmp[i] = self[idx][i]
    return(tmp)
  '''
  def rot_axis(self) :
    axis = VECTOR()
    co=(self[0][0]+self[1][1]+self[2][2]-1.0)/2.0;
    if co <= -1.0 :
      angl = pi;
      tmp=min(1.0,max(0,(self[0][0] + 1.0)/2.0))
      axis[0] = sqrt(tmp)
      tmp=min(1.0,max(0,(self[1][1] + 1.0)/2.0))
      axis[1] = sign(self[0][1])*sqrt(tmp)
      tmp=min(1.0,max(0,(self[2][2] + 1.0)/2.0))
      axis[2] = sign(self[0][2])*sqrt(tmp)
      si=0.0
    elif co < 1.0 :
      axis[0] = self[2][1] - self[1][2]
      axis[1] = self[0][2] - self[2][0]
      axis[2] = self[1][0] - self[0][1]
      an = abs(axis)
      if(an != 0.0) :
        for i in [0,1,2] :
          axis[i] = axis[i]/an
          si = an/2.0
        angl = atan2(si,co);
      else :
        angl = 0.0
        axis[0] = 1.0
        axis[1] = 0.0
        axis[2] = 0.0
    else :
      angl = 0.0
      axis[0] = 1.0
      axis[1] = 0.0
      axis[2] = 0.0
    return([angl,axis])
  '''
  
  def rot_axis(self) : # exraction of rotation angle and axis
    tmp=self.quaternion()
    co = tmp.w
    si = abs(tmp.v)
    th = 2*atan2(si,co)
    axis=[]
    if si != 0 :
      axis=1/si*tmp.v
    else :
       axis=VECTOR(1.0,0.0,0.0)
    return [th,axis]
 
  def quaternion(self):  # quaternion for rotation
    '''
    w=sqrt(self[0][0]+self[1][1]+self[2][2]+1)/2
    x=sign(self[2][1]-self[1][2])*sqrt(self[0][0]-self[1][1]-self[2][2]+1)/2
    y=sign(self[0][2]-self[2][0])*sqrt(-self[0][0]+self[1][1]-self[2][2]+1)/2
    z=sign(self[1][0]-self[0][1])*sqrt(-self[0][0]-self[1][1]+self[2][2]+1)/2
    '''
    tmp=self[0][0]+self[1][1]+self[2][2]+1
    if tmp < 0.0 :
       # print('w:',tmp)
       w = 0.0
    else :
       w=sqrt(tmp)/2
    tmp=self[0][0]-self[1][1]-self[2][2]+1
    if tmp < 0.0 :
       # print('x:',tmp)
       x = 0.0
    else :
       x=sign(self[2][1]-self[1][2])*sqrt(tmp)/2
    tmp=-self[0][0]+self[1][1]-self[2][2]+1
    if tmp < 0.0 :
       # print('y:',tmp)
       y = 0.0
    else :
       y=sign(self[0][2]-self[2][0])*sqrt(tmp)/2
    tmp=-self[0][0]-self[1][1]+self[2][2]+1
    if tmp < 0.0 :
       # print('z:',tmp)
       z = 0.0
    else :
       z=sign(self[1][0]-self[0][1])*sqrt(tmp)/2    
    return QUATERNION([w,VECTOR(x,y,z)])
    
  def abc(self) : # extraction of alpha, beta, gamma, rotation around x,y,z
    if self[0][2]>= 1.0 :
      b = pi/2;
      a = 0.0;
      c = atan2(self[2][1],self[1][1])
    elif self[0][2] <= -1.0 :
      b = -pi/2;
      a = 0.0;
      c = atan2(self[1][0],self[2][0])
    else :
      b = asin(self[0][2]);
      a = atan2(- self[1][2],self[2][2]);
      c = atan2(- self[0][1],self[0][0]);
    return([a, b, c])
  def rpy(self) : # extraction of roll, pitch, yaw, rotation around x,y,z
    return([-x for x in self.trans().abc()])
  def trans(self) : # transposition
    tmp = MATRIX()
    for i in [0,1,2] :
      for j in [0,1,2] :
        tmp[i][j] = self[j][i]    
    return(tmp)
  def __neg__(self) : # inverse, ie., transposition
    return(self.trans())
  def __mul__(self, other) : # multiplication with MATRIX or VECTOR
    tmp = None
    if isinstance(other, MATRIX) :
      tmp = MATRIX()
      for i in [0,1,2] :
        for j in [0,1,2] :
          tmp[i][j] = (self[i][0] * other[0][j] +
                       self[i][1] * other[1][j] +
                       self[i][2] * other[2][j])
    elif isinstance(other, VECTOR) :
      tmp = VECTOR()
      for i in [0,1,2] :
        tmp[i] = (self[i][0] * other[0] +
                  self[i][1] * other[1] +
                  self[i][2] * other[2])
    return(tmp)
  def __repr__(self) :
    return("m:" + list.__repr__(self))
  def __init__(self, mat=[], a=None, b=None, c=None, angle=None, axis=None) :
    # a,b,c are rotations around x,y,z axes
    list.__init__(self)
    self.append([1.0, 0.0, 0.0])
    self.append([0.0, 1.0, 0.0])
    self.append([0.0, 0.0, 1.0])
    if mat :
      for i in [0,1,2] :
        for j in [0,1,2] :
          self[i][j] = mat[i][j]
    elif a  :
      self[0][0] = 1.0;
      self[1][1] = cos(a);
      self[2][2] = cos(a);
      self[1][2] = -sin(a);
      self[2][1] = sin(a);
    elif b  :
      self[1][1] = 1.0;
      self[2][2] = cos(b);
      self[0][0] = cos(b);
      self[2][0] = -sin(b);
      self[0][2] = sin(b);
    elif c  :
      self[2][2] = 1.0;
      self[0][0] = cos(c);
      self[1][1] = cos(c);
      self[0][1] = -sin(c);
      self[1][0] = sin(c);
    elif axis :
      len = abs(axis)
      if len != 0.0 and angle != None :
        atmp = (1.0/len) * axis
        co = cos(angle)
        si = sin(angle)

        self[0][0] = atmp[0]*atmp[0]*(1.0 - co) + co
        self[1][0] = atmp[1]*atmp[0]*(1.0 - co) + atmp[2]*si
        self[2][0] = atmp[2]*atmp[0]*(1.0 - co) - atmp[1]*si
        self[0][1] = atmp[0]*atmp[1]*(1.0 - co) - atmp[2]*si
        self[1][1] = atmp[1]*atmp[1]*(1.0 - co) + co
        self[2][1] = atmp[2]*atmp[1]*(1.0 - co) + atmp[0]*si
        self[0][2] = atmp[0]*atmp[2]*(1.0 - co) + atmp[1]*si
        self[1][2] = atmp[1]*atmp[2]*(1.0 - co) - atmp[0]*si
        self[2][2] = atmp[2]*atmp[2]*(1.0 - co) + co

class QUATERNION(object) : # quaternion mainly for rotation
  def __init__(self,w_v=None, a=None, b=None, c=None, angle=None, axis=None) :
    if w_v :
      self.w=w_v[0]       
      self.v=VECTOR(vec=w_v[1])
    else :
      if a != None :
        angle=a
        axis=VECTOR(1,0,0)
      elif b != None :
        angle = b
        axis=VECTOR(0,1,0)
      elif c != None  :
        angle = c
        axis=VECTOR(0,0,1)
      if angle != None :
        self.w = cos(angle/2)
        if axis != None :
          self.v = sin(angle/2)*axis.normalize()
        else :
          self.v = VECTOR(sin(angle/2),0,0)
      else :
        self.w = 1.0
        self.v = VECTOR()

  def __repr__(self) :
     return("q:("+ str(self.w) +", "+ repr(self.v) + ")")
  '''    
  def __mul__(self, other) :
    w=self.w*other.w-self.v.dot(other.v)
    v=self.w*other.v+other.w*self.v+self.v*other.v
    return QUATERNION([w,v])
  '''
  def __mul__(self, other) :
    w=self.w*other.w-self.v.dot(other.v)
    x=self.w*other.v[0]+other.w*self.v[0]+self.v[1]*other.v[2]-self.v[2]*other.v[1]
    y=self.w*other.v[1]+other.w*self.v[1]+self.v[2]*other.v[0]-self.v[0]*other.v[2]
    z=self.w*other.v[2]+other.w*self.v[2]+self.v[0]*other.v[1]-self.v[1]*other.v[0]
    return QUATERNION([w,VECTOR(x,y,z)])
  def __neg__(self) :
    return QUATERNION([-self.w, -self.v])
  def __abs__(self) :
    return(sqrt(self.w*self.w+self.v.dot(self.v)))
  def normalize(self) :
    tmp=abs(self)
    return QUATERNION([self.w/tmp,1.0/tmp*self.v])
  def inv(self) :
    tmp=abs(self)
    return QUATERNION([self.w/tmp, -1.0/tmp*self.v])
  def conjugate(self) :
    return QUATERNION([self.w, -self.v])
  def matrix(self) :
    tmp = MATRIX()
    tmp1 = self.w*self.w
    tmp2 = self.v[0]*self.v[0]
    tmp3 = self.v[1]*self.v[1]
    tmp4 = self.v[2]*self.v[2]
    tmp5 = self.w*self.v[0]
    tmp6 = self.w*self.v[1]
    tmp7 = self.w*self.v[2]
    tmp8 = self.v[0]*self.v[1]
    tmp9 = self.v[1]*self.v[2]
    tmp10 = self.v[2]*self.v[0]
    tmp[0][0]=tmp1+tmp2-tmp3-tmp4
    tmp[1][0]=2*(tmp8+tmp7)
    tmp[2][0]=2*(tmp10-tmp6)
    tmp[0][1]=2*(tmp8-tmp7)
    tmp[1][1]=tmp1-tmp2+tmp3-tmp4
    tmp[2][1]=2*(tmp9+tmp5)
    tmp[0][2]=2*(tmp10+tmp6)
    tmp[1][2]=2*(tmp9-tmp5)
    tmp[2][2]=tmp1-tmp2-tmp3+tmp4
    '''                        
    tmp[0][0]=self.w*self.w+self.v[0]*self.v[0]-self.v[1]*self.v[1]-self.v[2]*self.v[2]
    tmp[1][0]=2*(self.v[1]*self.v[0]+self.w*self.v[2])
    tmp[2][0]=2*(self.v[2]*self.v[0]-self.w*self.v[1])
    tmp[0][1]=2*(self.v[0]*self.v[1]-self.w*self.v[2])
    tmp[1][1]=self.w*self.w-self.v[0]*self.v[0]+self.v[1]*self.v[1]-self.v[2]*self.v[2]
    tmp[2][1]=2*(self.v[2]*self.v[1]+self.w*self.v[0])
    tmp[0][2]=2*(self.v[0]*self.v[2]+self.w*self.v[1])
    tmp[1][2]=2*(self.v[1]*self.v[2]-self.w*self.v[0])
    tmp[2][2]=self.w*self.w-self.v[0]*self.v[0]-self.v[1]*self.v[1]+self.v[2]*self.v[2]
    '''
    return tmp

class FRAME(object) :
  def xyzabc(self) :
    tmp = self.mat.abc()
    return([self.vec[0],self.vec[1],self.vec[2],
            tmp[0],tmp[1],tmp[2]])
  def xyzrpy(self) :
    tmp = self.mat.rpy()
    return([self.vec[0],self.vec[1],self.vec[2],
            tmp[0],tmp[1],tmp[2]])
  def __neg__(self) :
    return(FRAME(mat=(-self.mat),vec=(-self.mat)*(-self.vec)))
  def __mul__(self, other) :
    tmp = None
    if isinstance(other, FRAME) :
      tmp_mat = self.mat * other.mat
      tmp_vec = (self.mat * other.vec) + self.vec
      tmp = FRAME(mat=tmp_mat, vec=tmp_vec)
    elif isinstance(other, VECTOR) :
      tmp = (self.mat * other) + self.vec
    return(tmp)
  def __repr__(self) :
    return("f:(" + repr(self.mat) + "," + repr(self.vec) + ")")
  def __init__(self, frm=[], mat=[], vec=[], xyzabc=[], xyzrpy=[]) :
    if frm :
      self.mat = MATRIX(mat=frm.mat)
      self.vec = VECTOR(vec=frm.vec)
    elif xyzabc :
      self.vec = VECTOR(xyzabc[0],xyzabc[1],xyzabc[2])
      tmp_a = MATRIX(a=xyzabc[3])
      tmp_b = MATRIX(b=xyzabc[4])
      tmp_c = MATRIX(c=xyzabc[5])
      self.mat = tmp_a * tmp_b * tmp_c
    elif xyzrpy :
      self.vec = VECTOR(xyzrpy[0],xyzrpy[1],xyzrpy[2])
      tmp_r = MATRIX(a=xyzrpy[3])
      tmp_p = MATRIX(b=xyzrpy[4])
      tmp_y = MATRIX(c=xyzrpy[5])
      self.mat = tmp_y * tmp_p * tmp_r
    else :
      self.mat = MATRIX(mat=mat)
      self.vec = VECTOR(vec=vec)
