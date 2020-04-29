from math import log2, pi
from matplotlib.mlab import specgram
from numpy import amax, argmax, arange, concatenate, interp, prod, zeros
from numpy.random import randn
from scipy.signal import hann, sawtooth

def a1 (x, fm, plim=None, dt=None, n=None, tw=None, dlog2p=None, s0=None):
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
    log2pc = arange(log2(plim[0]), log2(plim[1]), dlog2p) # log2 de los candidatos a altura
    pc = 2**log2pc # candidatos a altura
    ws = 2**round(log2(tw*fm)) # tamano de la ventana (muestras)
    w = hann(ws); # ventana Hann de tamano ws
    o = ws/2; # traslape entre las ventanas (recomendado para ventana Hann: 50%)
    t = arange(0, len(x)/fm, dt) # tiempos de los estimados de altura
    x = concatenate((zeros(ws//2), x, zeros(ws//2))) # se agregan ceros al principio para garantizar que ventanas cubran principio y final de la senal
    S = zeros((len(pc), len(t))) # matriz de puntos de los candidatos a lo largo del tiempo
    (X,f,tj) = specgram(x, ws, fm, window=w, noverlap=o, mode='magnitude') # calculo del espectro; devuelve tambien frecuencias y tiempos 
    for i in range(0, len(pc)): # para cada candidato:
        fi = pc[i]*arange(1,n+1) # identifique primeras n armonicas
        A = zeros((len(fi), X.shape[1])) # cree matriz nArmonicas x nVentanas
        for j in range(0, X.shape[1]): # por cada ventana:
            A[:,j] = interp(fi, f, X[:,j]) # estime el espectro en las armonicas
        si = prod(A, 0) # calcule el puntaje del candidato en cada ventana multiplicando el valor del espectro en las armonicas
        S[i,:] = interp(t, tj, si) # interpole los puntajes a los tiempos deseados
    p = pc[argmax(S,0)] # escoja el candidato con mayor puntuacion en cada ventana
    s = amax(S,0) # registre la puntuacion obtenida
    p[s < s0] = float('nan') # # indefina alturas cuyo puntaje no excede el umbral    
    return (p, t, s, pc, S) # devuelva ganadores, tiempos, puntajes, lista de candidatos y matriz de puntajes
