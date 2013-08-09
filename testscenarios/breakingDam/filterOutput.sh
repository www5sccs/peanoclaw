#!/bin/bash

CreateCell0="Vertex. after creating cell at \[0.703704,0.567901\]"
CreateCell1="Vertex. after creating cell at \[0.691358,0.567901\]"
CreateCell2="Vertex. after creating cell at \[0.703704,0.555556\]"
CreateCell3="Vertex. after creating cell at \[0.691358,0.555556\]"

DestroyCell0="Destroying cell.*\[0.703704,0.567901\]"
DestroyCell1="Destroying cell.*\[0.691358,0.567901\]"
DestroyCell2="Destroying cell.*\[0.703704,0.555556\]"
DestroyCell3="Destroying cell.*\[0.691358,0.555556\]"

SettingHanging="Setting hanging.*\[0.703704,0.567901\]"
SettingPersistent="Setting persistent vertex.*x:\[0.703704,0.567901\]"
SettingCellDescription="Setting cellDescriptionIndex\(.*\) from 2688 to 264"

egrep -n --color=auto "$CreateCell0|$CreateCell1|$CreateCell2|$CreateCell3|$DestroyCell0|$DestroyCell1|$DestroyCell2|$DestroyCell3|$SettingHanging|$SettingPersistent" output

#egrep -n "Vertex. after creating cell at \[0.6913.*,0.567901\]|Vertex. after creating cell at \[0.703704,0.567901\]|Setting persistent vertex.*x:\[0.703704,0.567901\]|Setting hanging.*\[0.703704,0.567901\]|Setting cellDescriptionIndex\(.*\) from 2688 to 264|Destroying cell.*\[0.691358,0.567901\]|Destroying cell.*\[0.703704,0.567901\]" output
