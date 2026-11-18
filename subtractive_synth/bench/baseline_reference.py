"""Pre-optimization reference capture (run against the ORIGINAL code).

This is the harness that produced ``output/baseline/``.  It is kept in
the repo so the before/after comparison is reproducible: check out the
pre-optimization commit, run this, then run ``bench/voice_bench.py`` on
the optimized code and diff the two ``summary.json`` files.  It uses the
exact same scenarios and the same measurement code as the optimized
benchmark.
"""

import json, os, sys, time
import numpy as np
import synth as _synth
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synth import SynthEngine, SynthParams, NoteEvent, SAMPLE_RATE, BLOCK_SIZE
from synth.config import WAVE_SAW
from synth.render import save_wav, measure_aliasing, spectrum, peak_stats, estimate_fundamental
from synth.oscillator import midi_to_freq, Oscillator

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "baseline")
SR=SAMPLE_RATE
os.makedirs(OUT, exist_ok=True)

def scenario(name, nvoices, notes, seconds=1.5, waveform=WAVE_SAW):
    p=SynthParams(voices=nvoices, waveform=waveform, attack=0.005, decay=0.12,
                  sustain=0.7, release=0.35, cutoff=6000.0, resonance=0.9,
                  gain=0.8, smooth_time=0.02)
    events=[NoteEvent(0.10, m, 0.9, "on") for m in notes]
    events+=[NoteEvent(1.10, m, 0.0, "off") for m in notes]
    eng=SynthEngine(p, SR, BLOCK_SIZE)
    block_ms=[]; audio=np.zeros(int(seconds*SR))
    total=int(seconds*SR)
    ev=sorted(events,key=lambda e:e.time)
    for s in range(0,total,BLOCK_SIZE):
        n=min(BLOCK_SIZE,total-s)
        t0,t1=s/SR,(s+n)/SR
        be=[e for e in ev if t0<=e.time<t1]
        t=time.perf_counter()
        audio[s:s+n]=eng.process_block(n,be,s)
        block_ms.append((time.perf_counter()-t)*1000)
    save_wav(os.path.join(OUT,name+".wav"),audio,SR)
    active=len([v for v in eng.voices if v.note is not None and not v.idle])
    # steady segment spectrum (0.4..0.9s)
    seg=audio[int(0.45*SR):int(0.95*SR)]
    fr,mag,db=spectrum(seg,SR,nfft=2**16)
    res={
      "voices_configured":nvoices,"notes":notes,"waveform":waveform,
      "params":vars(p),
      "active_after_note_off":active,
      "peak":peak_stats(audio,SR),
      "per_block_ms":{
        "blocks":len(block_ms),"mean":round(float(np.mean(block_ms)),4),
        "p95":round(float(np.percentile(block_ms,95)),4),
        "max":round(float(np.max(block_ms)),4),
        "deadline":round(BLOCK_SIZE/SR*1000,3),
        "mean_pct_deadline":round(100*np.mean(block_ms)/(BLOCK_SIZE/SR*1000),2),
      },
      "spectrum_peak_db":round(float(db.max()),2),
    }
    # measure aliasing of the first (lowest) note's fundamental in isolation
    f0=midi_to_freq(notes[0]); res["isolated_alias"]=measure_aliasing(seg,f0,SR)
    # verify every fundamental present
    df=fr[1]-fr[0]; floor=np.median(mag)
    present={str(m): bool(mag[int(round(midi_to_freq(m)/df))] > 10*floor) for m in notes}
    res["all_fundamentals_present"]=all(present.values()); res["fundamentals"]=present
    with open(os.path.join(OUT,name+".json"),"w") as f: json.dump(res,f,indent=2)
    return res, audio, fr, db

cases=[
 ("dense_chord_8",8,list(range(48,56))),
 ("dense_chord_16",16,list(range(48,64))),
 ("high_register_8",8,list(range(84,92))),
 ("high_register_16",16,list(range(84,100))),
]
summary={}
specs={}
for name,nv,notes in cases:
    res,audio,fr,db=scenario(name,nv,notes)
    summary[name]=res; specs[name]=(fr,db)
    print(name, "mean_ms",res["per_block_ms"]["mean"],"active",res["active_after_note_off"],
          "fund",res["all_fundamentals_present"],"alias",res["isolated_alias"]["alias_to_harmonic_db"])

# single high note aliasing (B7) both band-limited + naive reference
f0h=midi_to_freq(107)
band=Oscillator(WAVE_SAW,SR).render(np.full(int(1.5*SR),f0h))
alias=measure_aliasing(band,f0h,SR)
summary["single_note_B7"]={"f0":f0h,"alias":alias,"peak":peak_stats(band,SR)}

fig,ax=plt.subplots(figsize=(10,4))
for name,(fr,db) in specs.items():
    ax.plot(fr,db,label=name,lw=0.7)
ax.set_xlim(0,SR/2); ax.set_ylim(-120,5); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
ax.set_title("BASELINE spectra (pre-optimization)"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"spectra_baseline.png"),dpi=110); plt.close(fig)
summary["_provenance"]={"synth_module":os.path.abspath(_synth.__file__),
                        "python":sys.version.split()[0]}
with open(os.path.join(OUT,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
print("saved baseline artifacts")
