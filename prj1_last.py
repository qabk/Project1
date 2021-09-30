from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image   
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from tkinter.filedialog import askopenfilename
link = 0
lbl1 = 0


with open('face_label','rb') as f:
    face_label = pickle.load(f)
with open('face_name','rb') as f:
    index_name = pickle.load(f)
with open('predict','rb') as f:
    neigh = pickle.load(f)
image_tk = 0
image_tk1 = 0
fname1 = 0
import dlib
face_model = DeepFace.Facenet.loadModel()
hog_face_detector = dlib.get_frontal_face_detector()
x = 0
def face_reg(img):
    global neigh
    global face_model
    global face_vector_list
    global index_name
    global face_label
   
    
    

def face_reg():
    global image_tk1
    global fname1
    global x
    global neigh
    global face_model
    global index_name
    global face_label
    fname1 = np.array(fname1)
    faces_hog = hog_face_detector(fname1, 1)
    if(faces_hog == 0):
        return img
    for face in faces_hog :
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(fname1, (x,y), (x+w,y+h), (0,255,0), 2)
        vec = fname1[y:y+h,x:x+w]
        vec = cv2.resize(src = vec, dsize = (160,160))
        vec = vec.reshape(1,160,160,3).astype(np.double)/255
        vec = face_model.predict(vec)
        vec = vec/np.linalg.norm(vec)
        u, v = neigh.kneighbors(vec)
        v = v[u < 0.85]
        
        if(v.shape[0] < 3):
            cv2.putText(fname1,'unkown',(int(x),int(y-10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(0,0,255),1)
        
        else:
            predict_list = face_label[v]
            unique_elements, counts_elements = np.unique(predict_list, return_counts=True)
            #print(unique_elements.shape)
            #print(counts_elements)
            if(np.max(counts_elements) < 3 ):
                cv2.putText(fname1,'unkown',(int(x),int(y-10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(0,0,255),1)
    
            else:
                a = np.where(counts_elements == np.max(counts_elements))
                
                cv2.putText(fname1,index_name[unique_elements[a[0][0]]][5:],(int(x),int(y-10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(0,0,255),1)
    fname1 =cv2.cvtColor(fname1,cv2.COLOR_RGB2BGR)
    fname1 = cv2.cvtColor(fname1,cv2.COLOR_BGR2RGB)
    fname1 = Image.fromarray(fname1)
    image_tk1 = ImageTk.PhotoImage(fname1)
    #print(index_name[index[0]])
    
def bilateralfilter_img():
    global image_tk1
    global fname1
    fname1 = np.array(fname1)
    fname1 =cv2.cvtColor(fname1,cv2.COLOR_RGB2BGR)
    fname1 = cv2.bilateralFilter(fname1,15, 75, 75)
    fname1 = cv2.cvtColor(fname1,cv2.COLOR_BGR2RGB)
    fname1 = Image.fromarray(fname1)
    image_tk1 = ImageTk.PhotoImage(fname1)
def get_new_link():
    global link 
    global image_tk
    global fname1
    global lbl1
    global image_tk1
    #global lbl1
    if (link == 0):
        link = askopenfilename()
        #print(link1 + '  if')
        link_open =Image.open(link)
        fname1 = link_open.copy()
        image_tk = ImageTk.PhotoImage(link_open)
        image_tk1 = ImageTk.PhotoImage(link_open)
    
    else:
        #print(link + '  else')
        lbl1.grid_forget()
        link = askopenfilename()
        link_open =Image.open(link)
        fname1 = link_open.copy()
        image_tk = ImageTk.PhotoImage(link_open)
def adjust_constrast(x,a = 0):
    factor = 259*(a+255)/(255*(259-a))
    arr = factor*(x.copy().astype(np.float)-128)+128
    arr = np.clip(arr, a_min= 0, a_max=255).astype(np.uint8)
    return arr
def showImage():
    global lbl1
    #btn1 = ttk.Button(c, text="load image", command=showImage)
    lbl1 = ttk.Label(c)
    lbl1.configure(image=image_tk)
    lbl1.grid(column=0, row=400, sticky=S, pady=50, padx= 50)
    btn1.configure(text = "load image!", command=showImage)
    
def adjust_constrat_img():
    global link
    global image_tk1
    global fname1
    fname1 = np.array(fname1)
    fname1 = adjust_constrast(fname1,256)
    fname1 = Image.fromarray(fname1)
    image_tk1 = ImageTk.PhotoImage(fname1)

def his_equal():
    global link
    global image_tk1
    global fname1
    fname1 = np.array(fname1)
    fname1 =cv2.cvtColor(fname1,cv2.COLOR_RGB2BGR)
    fname1 = cv2.cvtColor(fname1,cv2.COLOR_BGR2HSV)
    fname1[:,:,2] = cv2.equalizeHist(fname1[:,:,2])
    fname1 = cv2.cvtColor(fname1,cv2.COLOR_HSV2RGB)
    fname1 = Image.fromarray(fname1)
    image_tk1 = ImageTk.PhotoImage(fname1)
def showImage1(): 
    global lbl2
    lbl2 = ttk.Label(c)
    lbl2.configure(image = image_tk1)
    lbl2.grid(column= 400, row = 400, sticky = S, pady = 50, padx = 50)
    btn2.configure(text = "load new img!", command=showImage1)  
    print(lbl2)
def delete_image():
    lbl2.grid_forget()


root = Tk()   


c = ttk.Frame(root, padding=(100, 100, 12, 0))
c.grid(column=0, row=0, sticky=(N,W,E,S))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0,weight=1)


btn1 = ttk.Button(c, text="load image", command=showImage)

btn2 = ttk.Button(c, text="load new img", command=showImage1)

btn1.grid(column=0, row=400, sticky=S, pady=50, padx=50)

btn2.grid(column = 400, row = 400, sticky = S, pady = 50, padx= 50)

a1 = Button(root,text='chose pic',command=get_new_link,width = 15).place(x=0,y=0)
a2 = Button(root,text='Delete_adj_img',command=lambda:delete_image(),width = 15).place(x=0,y=30)
a3 = Button(root,text='adj_constrast',command=adjust_constrat_img,width = 15).place(x=0,y=60)
a4 = Button(root,text='equal_hist',command=his_equal,width = 15).place(x=0,y=90)
a5 = Button(root,text='filter',command=bilateralfilter_img,width = 15).place(x=0,y=120)
a6 = Button(root,text='face_reg',command=face_reg,width = 15).place(x=0,y=150)

root.mainloop()
