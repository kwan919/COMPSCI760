load('b021_3_12') % load in cwru data files
load('b021_3_48')
load('normal_3')

csvwrite('b021_3_12_DE.csv', X225_DE_time) % save drive end signals 
csvwrite('b021_3_48_DE.csv', X229_DE_time)
csvwrite('normal_3_48_DE.csv', X100_DE_time)





