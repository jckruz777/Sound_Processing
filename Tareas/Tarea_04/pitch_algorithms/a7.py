from math import log2, pi
from matplotlib.mlab import specgram
import numpy as np
from numpy.random import randn
from scipy.signal import hann, sawtooth

def from_f_to_mm(f):
    # Conversion de frecuencia a milimetro
    m = 5.7 * np.log2(1 + (f / 230))
    return m

def from_mm_to_f(m):
    # Conversion de milimetro a frecuencia
    f = (2**(m / 5.7) - 1) * 230
    return f

def get_f_array(mm_array):
    # Conversion de arreglo de milimetros a arreglo de frecuencias
    n = len(mm_array)
    result = np.zeros(n)
    for i in range(n):
        m = mm_array[i]
        f = from_mm_to_f(m)
        result[i] = f
    return result

def create_delta_f_array(fi, delta_m, n):
    # Creacion de arreglo de frecuencias separadas cada delta_m milimetros en la coclea
    mm_array = np.zeros(n)
    mi = from_f_to_mm(fi)
    mm_array[0] = mi
    for i in range(1, n):
        mi += delta_m
        mm_array[i] = mi
    f_array = get_f_array(mm_array)
    return f_array

def a7 (x, fm, plim=None, dt=None, n=None, tw=None, dlog2p=None, s0=None):
    if plim is None: # rango de los candidatos a altura
        plim = [50,800] 
    if dt is None: # tiempo entre estimados de altura
        dt = 0.010 
    if n is None: # cantidad de armonicas a analizar
        n = 7 
    if tw is None: # tamano de la ventana (segundos)
        tw = 0.050 
    if dlog2p is None: # distancia entre los candidatos (octavas)
        dlog2p = 1/48 
    if s0 is None: # umbral de claridad de altura
        s0 = 0
    log2pc = np.arange(log2(plim[0]), log2(plim[1]), dlog2p) # log2 de los candidatos a altura
    pc = 2**log2pc # candidatos a altura
    ws = 2**round(log2(tw*fm)) # tamano de la ventana (muestras)
    w = hann(ws); # ventana Hann de tamano ws
    o = ws/2; # traslape entre las ventanas (recomendado para ventana Hann: 50%)
    t = np.arange(0, len(x)/fm, dt) # tiempos de los estimados de altura
    x = np.concatenate((np.zeros(ws//2), x, np.zeros(ws//2))) # se agregan ceros al principio para garantizar que ventanas cubran principio y final de la senal
    S = np.zeros((len(pc), len(t))) # matriz de puntos de los candidatos a lo largo del tiempo
    (X,f,tj) = specgram(x, ws, fm, window=w, noverlap=o, mode='magnitude') # calculo del espectro; devuelve tambien frecuencias y tiempos 
    N = ws  # tamaño de la ventana en número de muestras
    delta_m = 0.1 # una muestra cada 0.1 mm
    cochlea_pc = create_delta_f_array(pc[0], delta_m, len(pc)) # vector de candidatos con separacion de 0.1 mm 
    for i in range(0, len(cochlea_pc)): # para cada candidato:
        i_lim = int(N * cochlea_pc[i] / (4 * fm))
        f_lim = (N/2)
        del_rows = ()
        for j in range(X.shape[0]): # para cada fila de la matriz del espectro
            if j <= i_lim: # verificar si la fila pertenece a una frecuencia menor que el limite inferior de la sumatoria 
                del_rows += (j,) # añadir indice de fila a una tupla
        spectrum = np.delete(X, del_rows, 0) # eliminar las filas innecesarias del espectro
        k = np.arange(i_lim, f_lim) # limites de la sumatoria
        factor = np.sqrt((N * cochlea_pc[i]) / (k * fm))
        cosine = np.cos(2 * np.pi * k * fm / (N * cochlea_pc[i]))
        #print("- Deleted rows: " + str(len(del_rows)))
        #print("- Spectrum new shape: " + str(spectrum.shape))
        A = np.zeros((len(k), X.shape[1])) # cree matriz nArmonicas x nVentanas
        for h in range(0, X.shape[1]): # por cada ventana:
            A[:,h] = factor * np.sqrt(spectrum[:, h]) * cosine 
        si = np.sum(A, 0) # calcule el puntaje del candidato en cada ventana sumando el valor del espectro en las armonicas
        S[i,:] = np.interp(t, tj, si) # np.interpole los puntajes a los tiempos deseados
    p = cochlea_pc[np.argmax(S,0)] # escoja el candidato con mayor puntuacion en cada ventana
    s = np.amax(S,0) # registre la puntuacion obtenida
    p[s < s0] = float('nan') # indefina alturas cuyo puntaje no excede el umbral    
    return (p, t, s, cochlea_pc, S) # devuelva ganadores, tiempos, puntajes, lista de candidatos y matriz de puntajes
