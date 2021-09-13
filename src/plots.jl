using ElectronDisplay,PlotlyJS

function plot(p::Plasma{T}, sv::ScatterVolume{T}, freqs) where {T<:Real}
    trc = scatter(x=freqs, y=pwrspec(p, sv, freqs), showlegend=true,
        name="Ne=$(p.ne) m<sup>-3</sup><br>Te=$(Int(p.te)) K<br>Ti=$(Int(p.ti)) K")
    lyo = Layout(;title="IS spectrum", xaxis_title="Frequency in Hz", yaxis_title="Power",
        legend_x=0.75)
    PlotlyJS.plot([trc], lyo)
end
