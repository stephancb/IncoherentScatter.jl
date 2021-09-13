__precompile__()

module IncoherentScatter

const ℜ(z)=Base.real(z)
const ℑ(z)=Base.imag(z)
const ℂ(r, i) = Base.complex(r, i)

using FFTW,SpecialFunctions,Polynomials,PhysicalConstants.CODATA2018,Unitful
using CUDA
using LoopVectorization,Tullio

export Ion,Plasma,CollisionalPlasma,ScatterVolume,pwrspec,autocor,autocor!

friedconte(z) = im*√(π)*erfcx(-im*z)

struct Faddeeva
    n::Int
    l::Float64
    a::Polynomial{Float64}
    function Faddeeva(n=16)
        m  = 2*n
        l  = √(n/√(2))

        theta = (-m+1:m-1)*π/m
        t = l*tan.(theta/2)
        f = [0; exp.(-t.^2).*(l*l .+ t.^2)]
        a = real.(fft(fftshift(f)))[2:n-1]/(2*m)   # no flip because of convention in polyval
        new(n, l, Polynomial(a))
    end
end

const deff = Faddeeva() 

"""
    faddeeva(z, f=deff)

   The Faddeeva function is defined as:

   w(z) = exp(-z^2) * erfc(-im*z)

   where erfc(x) is the complex complementary error function. Also,
   
   friedconte(z) = im * √(π) * faddeeva(z)

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

   [2] B.D. Fried, S.D. Conte, The plasma dispersion function. New York Academic Press, 1961.
"""
function faddeeva(z, f::Faddeeva=deff)
    if ℜ(z)==0      # purely imaginary input
        exp.(-z.^2).*erfc(ℑ(z))
    else               # complex valued input
        # make sure input in the upper half-plane (positive imag. values)
        zz = ℑ(z)<0 ? conj(z) : z
        yy = (f.l + im*zz)/(f.l - im*zz)
        w  = 2*polyval(f.a, yy)/(f.l - im*zz)^2 + (1/√(π))/(f.l - im*zz)
        # convert the upper half-plane results to the lower half-plane if necessary
        ℑ(z)<0 ? conj(2*exp(-zz^2) - w) : w
    end
end

const c  = ustrip(u"m/s", SpeedOfLightInVacuum)
const me = ustrip(u"kg",  ElectronMass)
const e  = ustrip(u"C",   ElementaryCharge)
const kB = ustrip(u"J/K", BoltzmannConstant)
const ϵ0 = ustrip(u"F/m", VacuumElectricPermittivity)
const amu= ustrip(u"kg",  AtomicMassConstant)

const fnormf = c*√(me/(2*kB))/2
const debyef = (4*π/(e*c))^2*ϵ0*kB

const mp = ustrip(u"kg",  ProtonMass)
const mn = ustrip(u"kg",  NeutronMass)
const sqrtmpme = √((mp+mn)/me/2)

"gfe(B in Tesla) -> electron gyrofrequency in Hz"
gfe(B) = e*B/me

"""
    ScatterVolume(tfeff, system, b, alpha)

has parameters of the scattering volume that are constant in a fit.
"""
mutable struct ScatterVolume{T<:Real}
    tfeff::T    # effective transmitter frequency in Hz = tf*cos(scattering angle)
    fnormc::T   # factor for frequency normalization f -> theta
    debyec::T   # Debye length correction factor

    system::T   # spectrum scale factor (depends on tx power, volume size, ... )

    b::T        # magnetic field strength in Tesla
    gfe::T      # electron gyrofrequency
    phiel::T    # normalized electron gyrofrequency
    alpha::T    # angle between b and scattering vector in rad
    sinfac::T   # correction factor for magnetic field
    cosal::T    #

    function ScatterVolume{T}(tfeff, system, b, alpha) where {T<:Real}
        fnormc = fnormf/tfeff
        debyec = debyef*tfeff^2
        gf     = gfe(b)
        phiel  = fnormc*gf
        sinfac = 0.5*(sin(alpha)/phiel)^2
        cosal  = cos(alpha)
        new(tfeff, fnormc, debyec, system, b, gf, phiel, alpha, sinfac, cosal)
    end
end
# ScatterVolume(tfeff::T, system, b, alpha) where {T<:Real} = ScatterVolume{T}(tfeff, system, b, alpha)
ScatterVolume(;tfeff::T=230e6, system=1e-6, b=50000e-9, alpha=deg2rad(30)) where {T<:Real} =
    ScatterVolume{T}(tfeff, system, b, alpha)

struct Ion{T<:Real}
    mass::T   # ion mass in amu
    srmr::T   # sqrt of mass ratio ion/electron
end
Ion(m=15.999) = Ion(m, √(m*amu/me))

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
    
    function Plasma{T}(ne::T, te::T, ti::T, s::ScatterVolume{T}, fractn, ions, nion) where {T<:Real}
        sqti    = √(ti)
        thetc   = s.fnormc/sqti
        tex     = exp(-te*s.sinfac)
        tefac   = 0.5/te + s.sinfac/tex
        fractn1 = one(ne) - fractn[1] - fractn[2]
        dpsidc  = thetc*ions[1].srmr
        cdtine  = ti*s.debyec + ne
        ym      = ti*ne + te*cdtine
        new(ne, te, ti, fractn1, fractn, ions, nion,
            s.system*ne, s.debyec/ne, √(te), sqti, thetc,
            tex, tefac, dpsidc, cdtine, ym)
    end
end
Plasma(ne::T, te::T, ti::T,
       s::ScatterVolume{T}, nion=1, ions=(Ion(16*one(ne)), Ion(31*one(ne)), Ion(one(ne))),
       fractn=(zero(ne),zero(ne))) where {T<:Real} =
    Plasma{T}(ne, te, ti, s, fractn, ions, nion)
Plasma(s::ScatterVolume{T}; ne::T, te::T, ti::T,
       nion=1, ions=(Ion(16*one(ne)), Ion(31*one(ne)), Ion(one(ne))),
       fractn=(zero(ne),zero(ne))) where {T<:Real} =
    Plasma{T}(ne, te, ti, s, fractn, ions, nion)

struct CollisionalPlasma{T<:Real}
    p::Plasma{T}
    ν::T
end
CollisionalPlasma(p; ν) = CollisionalPlasma(p, ν)

"""
    ionadm(theta, z=friedconte(theta))
-> unmagnetized ion admittance for real normalized frequency
"""
ionadm(theta::T, z::Complex{T}=friedconte(theta)) where {T<:Real} =
    ℂ(theta*ℑ(z), 1 + theta*ℜ(z))

"""
    diadt1(::Plasma, ionadm::Complex)
-> derivative with respect to Ti of the unmagnetized ion admittance, 1st ion, zero frequency 
"""
diadt1(p::Plasma{T}, ia::Complex{T}) where {T<:Real} = 
    ℂ(0, p.fractn1*p.dpsidc*(√(π)*ℜ(ia) - 2*p.ions[1].srmr))

"""
    diadt1(θ, ::Plasma, ionadm::Complex)
-> derivative with respect to Ti of the unmagnetized ion admittance, 1st ion, real normalized frequency 
"""
function diadt1(θ::T, p::Plasma{T}, ia::Complex{T}) where {T<:Real}
    r5ti = T(0.5)/p.ti
    p.fractn1*((θ^2/p.ti - r5ti)*ia + im*r5ti)
end

function diadc(theta::T, ia::Complex{T}=ionadm(theta)) where {T<:Real}
    th =1/theta    
    ℂ(ℜ(ia)^2 + ℑ(ia)*(1 - ℑ(ia)))*th + ℑ(ia)*(th - 2*theta) - th,
      ℜ(ia)*(theta + (ℑ(ia) - 1)*th)
end

"""
    ionadm(freq, ::Plasma)
-> unmagnetized ion admittance for real unnormalized frequency and plasma parameters 
"""
function ionadm(freq::T, p::Plasma{T}, s::ScatterVolume{T}) where {T<:Real}
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
function ionadm(theta::T, z::T=friedconte(theta)) where {T<:Complex}
    psizm = ℑ(theta)*(ℜ(z)^2 + ℑ(z)^2)
    yd    = 1 - ℑ(theta)*(2*ℑ(z) - psizm)
    ℂ(ℜ(theta)*(ℑ(z) - psizm)/yd, 1 + ℜ(theta)*ℜ(z)/yd)
end

"""
    ionadm(freq, ::CollisionalPlasma)
-> unmagnetized ion admittance for unnormalized frequency and collisional plasma parameters 
"""
function ionadm(freq::T, coll::CollisionalPlasma{T}, s::ScatterVolume{T}) where {T<:Real}
    fnormt = s.fnormc/coll.p.sqti
    thetai = fnormt*coll.p.ions[1].srmr*ℂ(freq, coll.ν)
    yi     = ionadm(thetai)
    for k in 2:coll.p.nion
        if k==2
            yi *= coll.p.fractn1
        end
        thetai = fnormt*coll.p.ions[k].srmr*ℂ(freq, coll.ν)
        yi    += coll.p.fractn[k-1]*ionadm(thetai)
    end
    yi
end

"""
    elecadm(theta, tex, z)
-> magnetized electron admittance for real normalized frequency
"""
function elecadm(theta::T, tex::T, z::Complex{T}=friedconte(theta)) where {T<:Real}
    tt = tex*theta
    ℂ(tt*imag(z), 1 + tt*real(z)), imag(z)
end

# helper function, this part of the spectrum code is the same for non-collisional and collisional
"""
    espec(freq, ::Plasma, ::ScatterVolume, yi::Complex)
-> power spectral density for non-zero real frequency
"""
function espec(freq::T, p::Plasma{T}, s::ScatterVolume{T}, yi::Complex{T}) where {T<:Real}
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
function pwrspec(freq::T, p::Plasma{T}, s::ScatterVolume{T}) where {T<:Real}
    yi = ionadm(freq, p, s)
    espec(freq, p, s, yi)
end

"""
    pwrspec(freq, ::CollisionalPlasma, ::ScatterVolume)
-> power spectral density for non-zero frequency and collisional plasma
"""
function pwrspec(freq::T, coll::CollisionalPlasma{T}, s::ScatterVolume{T}) where {T<:Real}
    yi = ionadm(freq, coll, s)
    espec(freq, coll.p, s, yi)
end

# helper function
function yn0(p::Plasma{T}, s::ScatterVolume{T}, yir::T) where {T<:Real}
    tineq  = p.sqti*p.ne^2*yir*p.ti
    # sqtec  = p.cdtine*p.sqte*√(π)/s.cosal2 #  divide by colal, not colal*colal:
    sqtec  = p.cdtine*p.sqte*√(π)/s.cosal # zi(1) in fortran code = sqrt(π) for real zero frequency
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

"""
    pwrspec(::Plasma, ::ScatterVolume, freq_itr)
-> power spectral density for iterable frequencies (with length)
"""
function pwrspec(p::Union{Plasma{T}, CollisionalPlasma{T}}, s::ScatterVolume{T}, freqs;
                 threads=true) where {T<:Real}
    if threads
        pwr = Vector{T}(undef, length(freqs))
        if freqs[1]==0
            rng = 2:length(freqs)
            pwr[1] = pwrspec(p, s)
        else
            rng = 1:length(freqs)
        end
        Threads.@threads for k ∈ rng
            pwr[k] = pwrspec(freqs[k], p, s)
        end
    else
        if freqs[1]==0
            pwr = vcat(pwrspec(p, s), @inbounds pwrspec.(freqs[2:end], [p], [s]))
        else
            pwr = @inbounds pwrspec.(freqs, [p], [s])
        end
    end
    return pwr
end

function yirc(coll::CollisionalPlasma, kion::Int)
    psiion = coll.p.dpsidc*coll.ν
    z      = friedconte(ℂ(zero(coll.p.ne), psiion))
    psiz   = 1 - psiion*ℑ(z)
    psizm  = coll.p.ions[kion].srmr/psiz
    psizm*ℑ(z)
end

"""
    pwrspec(::CollisionalPlasma)
-> power spectral density for zero frequency and collisional plasma
"""
function pwrspec(coll::CollisionalPlasma, s::ScatterVolume)
    yir = yirc(coll, 1)
    for k in 2:coll.p.nion
        if k==2
            yir *= coll.p.fractn1
        end
        z    = friedconte(ℂ(zero(coll.p.ne), p.thetc*coll.p.ions[k].srmr*coll.ν))
        yir += coll.p.fractn[k-1]*yirc(coll, k)
    end

    yn = yn0(coll.p, s, yir)
    s.fnormc*coll.p.spectrsf*yn/coll.p.ym^2
end

"""
    cosine(freqs, τ)
-> matrix for the transform spectrum->ACF, method by Wes Swartz, ISCATSPE.FOR
"""
function cosine(freqs, τ) 
    c           = Matrix{eltype(freqs)}(undef, length(freqs), length(τ))
    c[    1, 1] = freqs[2]/2
    for k ∈ 2:length(freqs)-1
        c[k, 1] = (freqs[k+1]-freqs[k-1])/2
    end
    c[end, 1] = (freqs[end] - freqs[end-1])/2
    df        = diff(freqs)
    for l ∈ 2:length(τ)
        τ2    = 2*π*τ[l]
        c[1, l]       = (1 - cos(τ2*freqs[2]))/(τ2*τ2*freqs[2])
        c2            =  cos.(τ2*freqs[2:end-1])
        c[2:end-1, l] = ((c2 - cos.(τ2*freqs[1:end-2]))./df[1:end-1] +
                         (c2 - cos.(τ2*freqs[3:end  ]))./df[2:end  ])/(τ2*τ2)
        cend = cos(τ2*freqs[end])
        fend = 2*freqs[end]-freqs[end-1]
        c[end, l]     =  (2*cend - cos(τ2*freqs[end-1]) - cos(τ2*fend))/(df[end]*τ2*τ2) 
    end
    return c
end

struct FT{T<:Real}
    cft::Matrix{T}
end
FT(freqs, τ) = FT(cosine(freqs, τ))

function pwr2acf(pwr::Vector{T}, ftm::Matrix{T}) where {T<:Real}
    @tullio acf[l] := ftm.cft[k, l]*pwr[k]
end

function autocor(cft::Matrix{T}, pwr::Vector{T}) where {T<:Real}
    @tullio acf[l] := cft[k, l]*pwr[k]
end

function autocor!(acf::Vector{T}, cft::Matrix{T}, pwr::Vector{T}) where {T<:Real}
    @tullio acf[l] = cft[k, l]*pwr[k]
end

greet() = print("Hello World!")

end   # module IncoherentScatter
