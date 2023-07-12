import argparse
import numpy as np
import chumpy as ch 
import pandas as pd 
import pickle
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument("--mano_model",  default = './MANO_RIGHT.pkl')
parser.add_argument("--handy_model", default = './Right_Hand_Shape_35_UV_Cropped.pkl')
parser.add_argument("--correspondence_file", default = './Mano_7k_Template_correspondence.npy')
parser.add_argument("--output_model", default = './HANDY_RIGHT.pkl')
args = parser.parse_args()


def get_deformation(verts, correspondence_dict):
    faces_idx  = correspondence_dict['faces_idx']
    vert_weights = correspondence_dict['weights']
    mano_faces = correspondence_dict['mano_faces']

    bsize = verts.shape[0]
    face_verts= verts[mano_faces]
    

    v0, v1, v2 = face_verts[:, 0], face_verts[:,  1], face_verts[ :,  2]
    a = v0[faces_idx]
    b = v1[faces_idx]
    c = v2[faces_idx]

    
    deformation = vert_weights[0][:, None] * a \
                + vert_weights[1][:, None] * b \
                + vert_weights[2][:, None] * c   
    return deformation


mano_model  = pickle.load(open(args.mano_model, 'rb'), encoding='latin1')
handy_model = pickle.load(open(args.handy_model, 'rb'))

    
correspondence_dict = np.load(args.correspondence_file, allow_pickle=True).item()

new_dict = mano_model.copy()

new_dict['shapedirs']  = ch.array(handy_model['components'].T.reshape(7231,3 , -1))
new_dict['v_template'] = handy_model['v_template']
new_dict['f']          = np.array(handy_model['f'])

new_dict['weights']    = get_deformation(mano_model['weights'], correspondence_dict) 
J_reg = get_deformation(mano_model['J_regressor'].T.toarray(), correspondence_dict)
J_reg = J_reg / J_reg.sum(0)

new_dict['J_regressor']= sparse.csr_matrix(J_reg).T

new_dict['posedirs']   = get_deformation(mano_model['posedirs'].reshape(778,-1 ), correspondence_dict).reshape(7231 ,3, 135)


with open(args.output_model, 'wb') as f:
    pickle.dump(new_dict, f)
