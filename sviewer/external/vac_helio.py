# vac_helio - Converts air wavelengths to vacuum wavelengths and
#             applies velocity heloicentric correction
#             Using formula of Edlen 1966
#             Borrowed from Srianand and Noterdaeme

def vac_helio(l, v_helio):

    n = 1.0
    for i in range(5):
        n_it = n
        sig2 = 1.0e8/(l*l*n_it*n_it)
        n=1.0e-8*(15997.0/(38.90-sig2)+2406030.0/(130.0-sig2)+8342.13)+1.0

    l_0 = l*n*(1.0+v_helio/299792.458)
    return l_0
