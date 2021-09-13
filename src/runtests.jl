using IncoherentScatter
using ElectronDisplay,PlotlyJS
using Tullio,LoopVectorization

import Base: Threads.@threads

function testspectra(;threads=true)
    sv = ScatterVolume(tfeff=230e6, system=1e-6, b=50000e-9, alpha=deg2rad(13));
    freqs = 0.0:20:6400

    tr = Vector{GenericTrace}(undef, 3)

    for k ∈ 1:3
        p     = Plasma(ne=1.5e11, te=(k+1)*500.0, ti=1000.0, sv)
        testr = "$(Int(p.te))"
        tr[k] = scatter(x=freqs, y=pwrspec(p, sv, freqs; threads), showlegend=true,
                        name="Te=$testr K")
    end
    lyo = Layout(;title="IS spectrum, varying Te/Ti<br>Ne=1.5*10<sup>11</sup>, Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))

    for k ∈ 1:3
        p     = Plasma(ne=1.5e11, te=1500.0, ti=1000.0, sv, nion=2, fractn=((k-1)*0.5, 0.0))
        ofstr = "$(Int(p.fractn1*100))"
        mfstr = "$(Int(p.fractn[1]*100))"
        tr[k] = scatter(x=freqs, y=pwrspec(p, sv, freqs), showlegend=true,
                        name="O<sup>+</sup> $ofstr %, M<sup>+</sup> $mfstr %")
    end
    lyo = Layout(;title="IS spectrum, varying composition<br>"*
                        "Ne=1.5*10<sup>11</sup>, Te=1500 K, Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))

    for k ∈ 1:3
        p     = Plasma(ne=1.0*10^(9+k), te=2000.0, ti=1000.0, sv)
        tr[k] = scatter(x=freqs, y=10^(3-k)*pwrspec(p, sv, freqs), showlegend=true,
                        name="scale 10<sup>$(3-k)</sup>, Ne=10<sup>$(9+k)</sup> m<sup>-3</sup>")
    end
    lyo = Layout(;title="IS spectrum, varying Debye length<br>Te=2000 K, Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))

    for k ∈ 1:3
        svv = ScatterVolume(tfeff=230e6, system=1e-6, b=50000e-9,
                                             alpha=deg2rad((k-1)*30.0))
        p   = Plasma(ne=1.5e11, te=2000.0, ti=1000.0, sv)
        tr[k] = scatter(x=freqs, y=pwrspec(p, svv, freqs), showlegend=true,
                        name="∠(k, B) = $((k-1)*30) deg")
    end
    lyo = Layout(;title="IS spectrum, varying magnetic field<br>"*
                        "Ne=1.5*10<sup>11</sup>, Te=2000 K Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))
    
    p = Plasma(ne=1.5e11, te=300.0, ti=300.0, sv)
    for k ∈ 1:3
        cp    = CollisionalPlasma(p; ν=(k-1)*300.0)
        cfstr = "$(Int(cp.ν))"
        tr[k] = scatter(x=freqs, y=pwrspec(cp, sv, freqs), showlegend=true, name="ν=$cfstr Hz")
    end
    lyo = Layout(;title="IS spectrum, ν<br>"*
                        "Ne=1.5*10<sup>11</sup>, Te=300 K Ti=300 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))
end

function testft()
    freqs = vcat(0.0:15:4800, 4850:50:7600)
    τ     = 0.0:2e-5:144e-5
    cft   = IncoherentScatter.cosine(freqs, τ)

    sv    = ScatterVolume(tfeff=230e6, system=1e-6, b=50000e-9, alpha=deg2rad(13));
    tr = Vector{GenericTrace}(undef, 3)

    for k ∈ 1:3
        p     = Plasma(ne=1.5e11, te=(k+1)*500.0, ti=1000.0, sv)
        acf   = autocor(cft, pwrspec(p, sv, freqs))
        nm    = "Te=$(Int(p.te)) K"
        tr[k] = scatter(x=τ, y=acf, mode="lines+markers", showlegend=true, name=nm)
    end
    lyo = Layout(;title="ACF, varying Te/Ti<br>Ne=1.5*10<sup>11</sup>, Ti=1000 K", 
        xaxis_title="Lag in s", yaxis_title="Power", legend_x=0.75)
    display(plot(tr, lyo))

    for k ∈ 1:3
        p     = Plasma(ne=1.5e11, te=1500.0, ti=1000.0, sv, nion=2, fractn=((k-1)*0.5, 0.0))
        nm    = "O<sup>+</sup> $(Int(p.fractn1*100)) %, M<sup>+</sup> %$(Int(p.fractn[1]*100)) %"
        acf   = autocor(cft, pwrspec(p, sv, freqs))
        tr[k] = scatter(x=freqs, y=acf, mode="lines+markers", name=nm)
    end
    lyo = Layout(;title="IS spectrum, varying composition<br>"*
                        "Ne=1.5*10<sup>11</sup>, Te=1500 K, Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))

    for k ∈ 1:3
        p     = Plasma(ne=1.0*10^(9+k), te=2000.0, ti=1000.0, sv)
        nm    = "scale 10<sup>$(3-k)</sup>, Ne=10<sup>$(9+k)</sup> m<sup>-3</sup>"
        acf   = 10^(3-k)*autocor(cft, pwrspec(p, sv, freqs))
        tr[k] = scatter(x=freqs, y=nm, mode="lines+markers", showlegend=true, name=nm)
    end
    lyo = Layout(;title="IS spectrum, varying Debye length<br>Te=2000 K, Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))

    for k ∈ 1:3
        svv   = ScatterVolume(tfeff=230e6, system=1e-6, b=50000e-9, alpha=deg2rad((k-1)*30.0))
        p     = Plasma(ne=1.5e11, te=2000.0, ti=1000.0, sv)
        acf   = autocor(cft, pwrspec(p, svv, freqs))
        nm    = "∠(k, B) = $((k-1)*30) deg"
        tr[k] = scatter(x=freqs, y=autocor(cft, pwrspec(p, svv, freqs)), mode="lines+markers",
                        showlegend=true, name=nm)
    end
    lyo = Layout(;title="IS spectrum, varying magnetic field<br>"*
                        "Ne=1.5*10<sup>11</sup>, Te=2000 K Ti=1000 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))
    
    p = Plasma(ne=1.5e11, te=300.0, ti=300.0, sv)
    for k ∈ 1:3
        cp    = CollisionalPlasma(p; ν=(k-1)*300.0)
        acf   = autocor(cft, pwrspec(cp, sv, freqs))
        nm    = "ν=$(Int(cp.ν)) Hz"
        tr[k] = scatter(x=freqs, y=acf, mode="lines+markers", showlegend=true, name=nm)
    end
    lyo = Layout(;title="IS spectrum, ν<br>"*
                        "Ne=1.5*10<sup>11</sup>, Te=300 K Ti=300 K", 
        xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    display(plot(tr, lyo))
end
