import numpy as np
import glob

images = glob.glob('calib/*.jpeg')

for fname in images:
    pass


# O resultado da calibracao deve ser guardado nessas variaveis
ret, mtx, dist, rvecs, tvecs = None, None, None, None, None

# Salvar a matriz e a distorcao
np.save("cameraMatrix", mtx)
np.save("distCoeffs", dist)
print(mtx)
print(dist)