rm -rf python2_noedit
mkdir python2_noedit
cd mldatasets
for i in ./*.py; do
    3to2 $i > patch;
    cat patch | patch $i -o ../python2_noedit/$i;
done
rm patch
cd ..
