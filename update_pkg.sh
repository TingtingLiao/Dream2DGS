pip uninstall -y diff-surfel-rasterization
rm -rf submodules/diff-surfel-rasterization/build
rm -rf submodules/diff-surfel-rasterization/diff_surfel_rasterization.egg-info
pip install -e submodules/diff-surfel-rasterization

pip uninstall -y simple-knn
rm -rf submodules/simple-knn/build
rm -rf submodules/simple-knn/simple_knn.egg-info
pip install -e submodules/simple_knn