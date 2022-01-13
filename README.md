# Simulation of Dynamical Network Mechanisms Underlying Mouse Sleep (Work in Progress)
Modification of the mathematical model surrounding the sleep dynamics of mice described in "Coupled Flip-Flop Model for REM Sleep Regulation in the Rat" by accounting for changes in sleep patterns due to optogenetic stimulation of a desired brain region. Code created with the help of Dr. Weber in the Weber Lab at the Perelman School of Medicine at the University of Pennsylvania.

Experimental and computational studies in rodent sleep have sought to understand the synaptic interactions underlying the neuronal regulation of rapid eye movement (REM) sleep. A recent computational model has demonstrated success in recreating rodent sleep state transitions via the coupling of two mutually inhibitory (MI) “flip-flop” models describing synaptic interactions between wake-promoting, non REM (NREM) sleep-promoting, REM sleep-promoting (REM-on), and REM sleep-inhibiting (REM-off) neuronal populations. While current experimental evidence favors models by which the alternation between NREM and REM sleep is governed by MI interactions, it is unclear whether these MI models can account for newly-identified features correlating the duration of a REM sleep period to its following NREM state duration. In this study, we present a modification to the dynamical network of this coupled flip-flop model to address these features.

# Figures
![](images/No_Optogenetic_Activation_no_Noise.png)  
Example hypnogram of mouse sleep states over a 5 hour period predicted by the mathematical model used (no optogenetic activation). Purple blocks, gray, and blue blocks represent wake, non-REM, and REM states respectively. Firing rates of REM-on vs. REM-off and Wake vs. Sleep populations, REM and sleep pressures, and the applied optogenetic activation peiod are included under the hypnogram (there is not activation period in this figure).  
<br/>
<br/>  

![](images/Optogenetic_Activation_no_Noise.png)  
Example hypnogram of mouse sleep states over a 5 hour period with a period of optogenetic activation of a REM-off population (suppressing REM). Activation period is shown in the bottom subplot of the figure.  
<br/>
<br/>  

![](images/No_Optogenetic_Activation_with_Noise.png)  
Example hypnogram of mouse sleep states over a 5 hour period with added noise and no optogenetic activation.  
<br/>
<br/>  

![](images/Optogenetic_Activation_with_Noise.png)  
Example hypnogram of mouse sleep states over a 5 hour period with added noise and a period of optogenetic activation of a REM-off population (suppressing REM).  
