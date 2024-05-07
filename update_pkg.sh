pip uninstall -y diff-surfel-rasterization
rm -rf submodules/diff-surfel-rasterization/build
rm -rf submodules/diff-surfel-rasterization/diff_gaussian_rasterization.egg-info
pip install -e submodules/diff-surfel-rasterization