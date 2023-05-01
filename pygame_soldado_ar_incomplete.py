import cv2
from cv2 import aruco

import numpy as np

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from math import *

import ctypes
import argparse

from objloader import load


class Soldier:
    def __init__(
        self,
        vbo: list[int],
        vbon: list[int],
        nvertsbody: list[int],
        vbog: int,
        vbong: int,
        nvertsgun: int,
        color: tuple[int, int, int],
    ):
        self.vbo = vbo
        self.vbon = vbon
        self.nvertsbody = nvertsbody
        self.vbog = vbog
        self.vbong = vbong
        self.nvertsgun = nvertsgun
        self.r = color[0] / 255
        self.g = color[1] / 255
        self.b = color[2] / 255

    def draw(self, velocidade: int, time: int):
        meshID = int((time) / velocidade) % len(self.vbo)

        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glColor3f(self.r, self.g, self.b)
        glRotatef(90, 1, 0, 0)
        DrawBuffer(self.vbo[meshID], self.vbon[meshID], self.nvertsbody[meshID])
        DrawBuffer(self.vbog, self.vbong, self.nvertsgun)

        self.pos = glGetFloatv(GL_MODELVIEW_MATRIX).T


def LoadMeshes(path, numberOfMeshes):
    meshes = []
    for i in range(1, numberOfMeshes + 1):
        meshes.append(load(path.format(id=i)))

    return meshes


def DrawBuffer(vbo, vbon, noOfVertices):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, vbon)
    glEnableClientState(GL_NORMAL_ARRAY)
    glNormalPointer(GL_FLOAT, 0, None)

    glDrawArrays(GL_TRIANGLES, 0, noOfVertices)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def intrinsic2Project(
    MTX: np.ndarray,
    width: float,
    height: float,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> tuple[np.ndarray, list]:
    """[Get ]
    Arguments:
        MTX {[np.array]} -- [The camera instrinsic matrix that you get from calibrating your chessboard]
        width {[float]} -- [width of viewport]]
        height {[float]} -- [height of viewport]
    Keyword Arguments:
        near_plane {float} -- [near_plane] (default: {0.01})
        far_plane {float} -- [far plane] (default: {100.0})
    Returns:
        [np.array] -- [1 dim array of project matrix]
    """
    P = np.zeros(shape=(4, 4), dtype=np.float32)

    fx, fy = MTX[0, 0], MTX[1, 1]
    cx, cy = MTX[0, 2], MTX[1, 2]

    P[0, 0] = 0
    P[0, 2] = 0

    P[1, 1] = 0
    P[1, 2] = 0

    P[2, 2] = 0
    P[2, 3] = 0

    P[3, 2] = 0

    P = P.flatten(order="F")
    fov = [0, 0, 0, 0]
    return P, fov


def extrinsic2ModelView(RVEC: np.ndarray, TVEC: np.ndarray):
    """[Get modelview matrix from RVEC and TVEC]
    Arguments:
        RVEC {[vector]} -- [Rotation vector]
        TVEC {[vector]} -- [Translation vector]
    """

    R, _ = cv2.Rodrigues(RVEC)

    # invert para colocar a camera apontando para o -z
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Transforma em vetor coluna
    TVEC = TVEC.reshape((3, 1))

    transform_matrix = Rx @ np.hstack((R, TVEC))

    # cria uma identidade 4x4
    M = np.eye(4)
    M[:3, :] = transform_matrix
    return M.flatten(order="F")


def DrawObject(
    bg_image,
    matrix_coefficients,
    distortion_coefficients,
    soldierRed: Soldier,
    soldierGreen: Soldier,
    aruco_size,
):
    # contabiliza o tempo para atualizar a mesh da vez
    fator_velocidade = 20
    time = pygame.time.get_ticks()

    # operations on the frame come here
    gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)  # Change grayscale

    # SEU CODIGO AQUI
    rvec, tvec = None, None
    model_matrix = extrinsic2ModelView(rvec, tvec)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPushMatrix()
    glLoadMatrixf(model_matrix)

    soldierRed.draw(fator_velocidade, time)

    glPopMatrix()


def DrawBackground(texture_background, bg_image, fov, far):
    height = bg_image.shape[0]
    width = bg_image.shape[1]
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glColor3f(1.0, 1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)

    # Defina as coordenadas onde a imagem de fundo deve ser colocada
    pz = 0
    pxr = 0
    pxl = 0
    pyt = 0
    pyb = 0

    # draw background
    glBindTexture(GL_TEXTURE_2D, texture_background)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, bg_image
    )

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(pxl, pyb, -pz)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(pxr, pyb, -pz)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(pxr, pyt, -pz)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(pxl, pyt, -pz)
    glEnd()


def CreateBuffers(vertices: list[list[float]]) -> list[int]:
    # aloca os buffers para desenho
    vbo = glGenBuffers(len(vertices))
    for i in range(len(vertices)):
        bufferdata = (ctypes.c_float * len(vertices[i]))(*vertices[i])  # float buffer
        buffersize = len(vertices[i]) * 4  # buffer size in bytes
        glBindBuffer(GL_ARRAY_BUFFER, vbo[i])
        glBufferData(GL_ARRAY_BUFFER, buffersize, bufferdata, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo


def CreateBuffer(vertices: list[list[float]]) -> int:
    # aloca os buffers para desenho
    vbo = glGenBuffers(1)
    bufferdata = (ctypes.c_float * len(vertices[0]))(*vertices[0])  # float buffer
    buffersize = len(vertices[0]) * 4  # buffer size in bytes
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, buffersize, bufferdata, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo


def main():
    pygame.init()

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--Video", help="Video file")
    parser.add_argument("-c", "--Camera", help="Which camera")
    args = parser.parse_args()

    width = 1280
    height = 720
    print(f"res: {width} x {height}")
    cap = None
    if args.Video:
        cap = cv2.VideoCapture(args.Video)
    elif args.Camera:
        print(args.Camera)
        cap = cv2.VideoCapture(int(args.Camera))
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, bg_image = cap.read()
    if not ret:
        print("Error readining frame!")
        return

    # inicializa a janela da aplicacao
    screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    clock = pygame.time.Clock()

    # Constroi a matriz de projecao
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    near = 0.1
    far = 1000.0

    try:
        matrix_coefficients = np.load("cameraMatrix.npy")
        distortion_coefficients = np.load("distCoeffs.npy")
    except FileNotFoundError:
        raise Exception(
            "N foi possivel carregar as matrizes de calibraco e distorcao. Tente executar camera_calib.py antes."
        )

    projectMatrix, fov = intrinsic2Project(
        matrix_coefficients, width, height, near, far
    )
    glMultMatrixf(projectMatrix)

    soldados = LoadMeshes("./Blender/soldado{id:}.obj", 15)
    vts = [x.vertices for x in soldados]
    nms = [x.normals for x in soldados]
    vbo = CreateBuffers(vts)
    vbon = CreateBuffers(nms)
    numberOfVerts = list(map(lambda x: len(x) // 3, vts))

    arma = LoadMeshes("./Blender/armanova{id:}.obj", 1)
    vtas = [x.vertices for x in arma]
    nmas = [x.normals for x in arma]
    vbog = CreateBuffer(vtas)
    vbong = CreateBuffer(nmas)
    numberOfVertsArma = len(vtas[0])

    soldierRed = Soldier(
        vbo, vbon, numberOfVerts, vbog, vbong, numberOfVertsArma, (255, 0, 0)
    )
    soldierGreen = Soldier(
        vbo, vbon, numberOfVerts, vbog, vbong, numberOfVertsArma, (0, 255, 0)
    )

    # create background texture and seta configuracoes
    texture_background = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_background)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glEnable(GL_DEPTH_TEST)

    while True:
        # Trata eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN and (
                event.key == pygame.K_q or event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1, 0, 0, 1)

        ret, bg_image = cap.read()
        if args.Video:
            pygame.time.delay(40)

        DrawObject(
            bg_image,
            matrix_coefficients,
            distortion_coefficients,
            soldierRed,
            soldierGreen,
            90,
        )
        DrawBackground(texture_background, bg_image, fov, far)

        # Faz o update
        pygame.display.flip()


if __name__ == "__main__":
    main()
