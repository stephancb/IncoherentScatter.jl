__precompile__()

module IScatspe

using SpecialFunctions,Polynomials

import PhysicalConstants.MKS: Boltzmann,ElectronCharge,MassElectron,SpeedOfLight,
                              UnifiedAtomicMass,VacuumPermittivity

friedconte(z) = im*sqrt(pi)*erfcx(-im*z)

struct Faddeeva
    n::Int
    l::Float64
    a::Poly{Float64}
    function Faddeeva(n=16)
        m  = 2*n
        l  = sqrt(n/sqrt(2))

        theta = (-m+1:m-1)*pi/m
        t = l*tan.(theta/2)
        f = [0; exp.(-t.^2).*(l*l + t.^2)]
        a = real.(fft(fftshift(f)))[2:n-1]/(2*m)   # no flip because of convention in polyval
        new(n, l, Poly(a))
    end
end

const deff = Faddeeva() 

"""
    faddeeva(z, f=deff)

   The Faddeeva function is defined as:

   w(z) = exp(-z^2) * erfc(-im*z)

   where erfc(x) is the complex complementary error function. Also,
   
   friedconte(z) = im * sqrt(pi) * faddeeva(z)

   where friedconte(z) is the plasma dispersion function as defined in [2].

   w = faddeeva(z, f) reuses previously calculated coefficients in struct f
   where f is initialized like f=Faddeeva(n) with n specifying the number of terms
   to truncate the expansion (see (13) in [1]). n = 16 is used as default, i.e. for f=Faddeeva().

   Example:
       x = linspace(-10,10,1001); [X,Y] = meshgrid(x,x); 
       W = faddeeva(complex(X,Y)); 

   Reference:
   [1] J.A.C. Weideman, "Computation of the Complex Error Function," SIAM
       J. Numerical Analysis, pp. 1497-1518, No. 5, Vol. 31, Oct., 1994 
       Available Online: http://www.jstor.org/stable/2158232

   [2] B.D. Fried, S.D. Conte, The plasma dispersion function. New York Acadenic Press, 1961.
"""
function faddeeva(z, f::Faddeeva=deff)
    if real(z)==0      # purely imaginary input
        exp.(-z.^2).*erfc(imag(z))
    else               # complex valued input
        # make sure input in the upper half-plane (positive imag. values)
        zz = imag(z)<0? conj(z):z
        yy = (f.l + im*zz)/(f.l - im*zz)
        w  = 2*polyval(f.a, yy)/(f.l - im*zz)^2 + (1/sqrt(pi))/(f.l - im*zz)
        # convert the upper half-plane results to the lower half-plane if necessary
        imag(z)<0? conj(2*exp(-zz^2) - w):w
    end
end

const fnormf = SpeedOfLight*sqrt(MassElectron/(2*Boltzmann))/2
const debyef = (4*π/ElectronCharge/SpeedOfLight)^2*VacuumPermittivity*Boltzmann

"gfe(b in T) -> electron gyrofrequency in Hz"
gfe(b) = ElectronCharge*b/MassElectron

"""
    ScatterVolume(tfeff, system, b, alpha)

has parameters of the scattering volume that are constant in a fit.
"""
struct ScatterVolume{T<:Real}
    tfeff::T    # effective transmitter frequency in Hz
    fnormc::T   # factor for frequency normalization f -> theta
    debyec::T   # Debye length correction factor

    system::T   # spectrum scale factor (depends on tx power, volume size, ... )

    b::T        # magnetic field strength in Tesla
    gfe::T      # electron gyrofrequency
    phiel::T    # normalized electron gyrofrequency
    alpha::T    # angle between b and scattering vector in rad
    sinfac::T   # correction factor for magnetic field
    cosal::T    #
    cosal2::T

    function ScatterVolume{T}(tfeff::T, system::T, b::T, alpha::T) where T<:Real
        fnormc = fnormf/tfeff
        debyec = debyef*tfeff^2
        gf     = gfe(b)
        phiel  = fnormc*gf
        sinfac = 0.5*(sin(alpha)/phiel)^2
        cosal  = cos(alpha)
        new(tfeff, fnormc, debyec, system, b, gf, phiel, alpha, sinfac, cosal, cosal*cosal)
    end
end
ScatterVolume(tfeff::T, system::T, b::T, alpha::T) where {T<:Real} =
    ScatterVolume{T}(tfeff, system, b, alpha)

struct Ion{T<:Real}
    mass::T   # ion mass in amu
    srmr::T   # sqrt of mass ratio ion/electron
end
Ion(m=15.999) = Ion(m, sqrt(m*UnifiedAtomicMass/MassElectron))

"""
Plasma(ne, te, ti, ::ScatterVolume, [ions, fractn]

has parameters of the plasma that are constant over all frequencies of the spectrum.
"""
struct Plasma{T<:Real}
    ne::T
    te::T
    ti::T
    fractn1::T
    fractn::NTuple{2, T}    # up to two more ions
    ions::NTuple{3, Ion{T}}
    nion::Int

    spectrsf::T
    debyen::T   # normalized Debye length correction factor
    sqte::T
    sqti::T
    thetc::T
    tex::T
    tefac::T
    dpsidc::T

    # parameter for spectrum at zero frequency:
    cdtine::T
    ym::T
    
    function Plasma{T}(ne::T, te::T, ti::T, s::ScatterVolume{T}, fractn, ions, nion) where T<:Real
        sqti    = sqrt(ti)
        thetc   = s.fnormc/sqti
        tex     = exp(-te*s.sinfac)
        tefac   = 0.5/te + s.sinfac/tex
        fractn1 = one(ne) - fractn[1] - fractn[2]
        dpsidc  = thetc*ions[1].srmr
        cdtine  = ti*s.debyec + ne
        ym      = ti*ne + te*cdtine
        new(ne, te, ti, fractn1, fractn, ions, nion,
            s.system*ne, s.debyec/ne, sqrt(te), sqti, thetc,
            tex, tefac, dpsidc, cdtine, ym)
    end
end
Plasma(ne::T, te::T, ti::T,
       s::ScatterVolume{T}, nion=1, ions=(Ion(16*one(ne)), Ion(31*one(ne)), Ion(one(ne))),
       fractn=(zero(ne),zero(ne))) where {T<:Real} =
           Plasma{T}(ne, te, ti, s, fractn, ions, nion)

struct CollisionalPlasma{T<:Real}
    p::Plasma{T}
    cf::T
end

"""
    ionadm(theta, z=friedconte(theta))
-> unmagnetized ion admittance for real normalized frequency
"""
ionadm{T<:Real}(theta::T, z::Complex{T}=friedconte(theta)) = Complex(theta*imag(z), 1 + theta*real(z))

"""
    ionadm(freq, ::Plasma)
-> unmagnetized ion admittance for real unnormalized frequency and plasma parameters 
"""
function ionadm{T<:Real}(freq::T, p::Plasma{T}, s::ScatterVolume{T})
    fnormt = s.fnormc/p.sqti
    thetai = fnormt*p.ions[1].srmr*freq
    yi     = ionadm(thetai)
    for k in 2:p.nion
        if k==2
            yi *= p.fractn1
        end
        thetai = fnormt*p.ions[k].srmr*freq
        yi    += p.fractn[k-1]*ionadm(thetai)
    end
    yi
end

"""
    ionadm(theta, z=friedconte(theta))
-> unmagnetized ion admittance for complex normalized frequency.
"""
function ionadm{T<:Complex}(theta::T, z::T=friedconte(theta))
    psizm = imag(theta)*(real(z)^2 + imag(z)^2)
    yd    = 1 - imag(theta)*(2*imag(z) - psizm)
    Complex(real(theta)*(imag(z) - psizm)/yd, 1 + real(theta)*real(z)/yd)
end

"""
    ionadm(freq, ::CollisionalPlasma)
-> unmagnetized ion admittance for unnormalized frequency and collisional plasma parameters 
"""
function ionadm{T<:Real}(freq::T, coll::CollisionalPlasma{T}, s::ScatterVolume{T})
    fnormt = s.fnormc/coll.p.sqti
    thetai = fnormt*coll.p.ions[1].srmr*Complex(freq, coll.cf)
    yi     = ionadm(thetai)
    for k in 2:coll.p.nion
        if k==2
            yi *= coll.p.fractn1
        end
        thetai = fnormt*coll.p.ions[k].srmr*Complex(freq, coll.cf)
        yi    += coll.p.fractn[k-1]*ionadm(thetai)
    end
    yi
end

"""
    elecadm(theta, tex, z)
-> magnetized electron admittance for real normalized frequency
"""
function elecadm{T<:Real}(theta::T, tex::T, z::Complex{T}=friedconte(theta))
    tt = tex*theta
    Complex(tt*imag(z), 1 + tt*real(z)), imag(z)
end

# helper function, this part of the spectrum code is the same for non-collisional and collisional
"""
    espec(freq, ::Plasma, ::ScatterVolume, yi::Complex)
-> power spectral density for non-zero real frequency
"""
function espec{T<:Real}(freq::T, p::Plasma{T}, s::ScatterVolume{T}, yi::Complex{T})
    yiti   = yi/p.ti + im*p.debyen

    # electron admittance:
    thetae = s.fnormc/(p.sqte*s.cosal)*freq
    ye,    = elecadm(thetae, p.tex)
    yete   = ye/p.te

    the2   = thetae^2
    texthe = p.tex*thetae

    ym     = abs2(yete+yiti)
    yn     = abs2(yete)*real(yi) + abs2(yiti)*real(ye)

    (p.spectrsf/freq)*(yn/ym)
end

"""
    pwrspec(freq, ::Plasma, ::ScatterVolume)
-> power spectral density for non-zero real frequency
"""
function pwrspec{T<:Real}(freq::T, p::Plasma{T}, s::ScatterVolume{T})
    yi = ionadm(freq, p, s)
    espec(freq, p, s, yi)
end

"""
    pwrspec(freq, ::CollisionalPlasma, ::ScatterVolume)
-> power spectral density for non-zero frequency and collisional plasma
"""
function pwrspec{T<:Real}(freq::T, coll::CollisionalPlasma{T}, s::ScatterVolume{T})
    yi = ionadm(freq, coll, s)
    espec(freq, coll.p, s, yi)
end

# helper function
function yn0{T<:Real}(p::Plasma{T}, s::ScatterVolume{T}, yir::T)
    tineq  = p.sqti*p.ne^2*yir*p.ti
    sqtec  = p.cdtine*p.sqte*sqrt(pi)/s.cosal2 # zi(1) in fortran code = sqrt(π) for real zero frequency
    stecx  = sqtec*p.tex*p.te
    tineq + stecx*p.cdtine
end

"""
    pwrspec(::Plasma)
-> power spectral density for zero real frequency
"""
function pwrspec(p::Plasma, s::ScatterVolume)
    yir = sqrt(π)*p.ions[1].srmr
    for k in 2:p.nion
        if k==2
            yir *= p.fractn1
        end
        yir += p.fractn[k-1]*sqrt(π)*p.ions[k].srmr
    end
    yn = yn0(p, s, yir)
    s.fnormc*p.spectrsf*yn/p.ym^2
end

function yirc(coll::CollisionalPlasma, kion::Int)
    psiion = coll.p.dpsidc*coll.cf
    z      = friedconte(Complex(zero(coll.p.ne), psiion))
    psiz   = 1 - psiion*imag(z)
    psizm  = coll.p.ions[kion].srmr/psiz
    psizm*imag(z)
end

"""
    pwrspec(::CollisionalPlasma)
-> power spectral density for zero frequency and collisional plasma
"""
function pwrspec(coll::CollisionalPlasma, s::ScatterVolume)
    yir = yirc(coll, 1)
    for k in 2:p.nion
        if k==2
            yir *= p.fractn1
        end
        z    = friedconte(Complex(zero(coll.p.ne), p.thetc*coll.p.ions[k].srmr*coll.cf))
        yir += p.fractn[k-1]*yirc(coll, k)
    end

    yn = yn0(p, s, yir)
    s.fnormc*coll.p.spectrsf*yn/p.ym^2
end

end   # module IScatspe
