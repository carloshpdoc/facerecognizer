import cv2
import os
import numpy as np 

eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=1.5)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('foto', f) for f in os.listdir('foto')]
    # print(caminhos)
    faces = []
    ids = []

    for caminhosImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhosImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhosImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
        # cv2.imshow('Face', imagemFace)
        # cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComId()
# print(ids)

print('treinamento começando')

eigenface.train(faces, ids)
eigenface.write('ClassificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('ClassificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('ClassificadorLBPH.yml')

print('treinamento realizado')