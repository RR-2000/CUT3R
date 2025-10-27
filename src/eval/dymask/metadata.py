allow_repeat=False
split='test'
ROOT_PO='/mnt/rdata4_6/kx_data/4d_dataset/point_odyssey'
ROOT_DAVIS='/home/ramanathan/data/DAVIS-2017'
resolution=(512, 384)
num_views=20
n_corres=0

metadata = {
    "PointOdyssey": f"100 @ PointOdyssey_Multiview(allow_repeat={allow_repeat}, split='test', ROOT='{ROOT_PO}', resolution={resolution}, num_views={num_views}, n_corres={n_corres})",
    "Davis-16": f"20 @ DAVIS(allow_repeat={allow_repeat}, version='2016', split='test', ROOT='{ROOT_DAVIS}', resolution={resolution}, num_views={num_views}, n_corres={n_corres})",
    "Davis-17": f"30 @ DAVIS(allow_repeat={allow_repeat}, version='2017', split='test', ROOT='{ROOT_DAVIS}', resolution={resolution}, num_views={num_views}, n_corres={n_corres})",
    "Davis-All": f"50 @ DAVIS(allow_repeat={allow_repeat}, version='all', split='test', ROOT='{ROOT_DAVIS}', resolution={resolution}, num_views={num_views}, n_corres={n_corres})",

}