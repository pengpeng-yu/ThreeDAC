import os.path as osp
import subprocess
from typing import Union

from plyfile import PlyData, PlyElement
import numpy as np
import torch


def write_ply_file(
        xyz: Union[torch.Tensor, np.ndarray],
        file_path: str,
        rgb: Union[torch.Tensor, np.ndarray]) -> None:
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    assert xyz.shape[1] == 3
    xyz_dtype = '<f4'
    xyz = xyz.astype(xyz_dtype)
    rgb_dtype = np.uint8
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    assert rgb.shape[1] == 3 and rgb.shape[0] == xyz.shape[0]
    assert rgb.dtype in (np.float32, rgb_dtype)
    rgb = rgb.astype(rgb_dtype)
    el_with_properties_dtype = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    el_with_properties_dtype.extend([('red', rgb_dtype), ('green', rgb_dtype), ('blue', rgb_dtype)])
    el_with_properties = np.empty(len(xyz), dtype=el_with_properties_dtype)
    el_with_properties['x'] = xyz[:, 0]
    el_with_properties['y'] = xyz[:, 1]
    el_with_properties['z'] = xyz[:, 2]
    el_with_properties['red'] = rgb[:, 0]
    el_with_properties['green'] = rgb[:, 1]
    el_with_properties['blue'] = rgb[:, 2]
    el = PlyElement.describe(el_with_properties, 'vertex')
    PlyData([el]).write(file_path)


class LogExtractor:
    def __init__(self):
        pass

    def extract_log(self, log: str, mappings):
        lines = log.splitlines()
        extracted = {}
        for key, (new_key, map_fn) in mappings.items():
            for idx, line in enumerate(lines):
                if line.startswith(key):
                    extracted[new_key] = map_fn(line)
                    lines = lines[idx + 1:]
                    break
        return extracted


class TMC3LogExtractor(LogExtractor):
    default_enc_log_mappings = {
        'colors bitstream size': ('bpp', lambda l: float(l.split()[-2][1:]))
    }

    def __init__(self):
        self.enc_log_mappings = self.default_enc_log_mappings
        super(TMC3LogExtractor, self).__init__()

    def extract_enc_log(self, log: str):
        return self.extract_log(log, self.enc_log_mappings)


log_extractor = TMC3LogExtractor()


_DIVIDERS = ['1. Use infile1 (A) as reference, loop over A, use normals on B. (A->B).',
             '2. Use infile2 (B) as reference, loop over B, use normals on A. (B->A).',
             '3. Final (symmetric).',
             'Job done!']


def mpeg_pc_error(
        infile1: str, infile2: str, resolution: int, normal_file: str = '',
        hausdorff: bool = False, color: bool = False, threads: int = 1
):
    cmd_args = f'bin/pc_error' \
               f' -a {infile1}' \
               f' -b {infile2}' \
               f' --resolution={resolution - 1}' \
               f' --hausdorff={int(hausdorff)}' \
               f' --color={int(color)}' \
               f' --nbThreads={threads}'
    if normal_file != '' and osp.exists(normal_file):
        cmd_args += ' -n ' + normal_file

    subp_stdout = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    ).stdout

    metric_dict = {}
    flag_read = False
    for line in subp_stdout.splitlines():
        if line.startswith(_DIVIDERS[0]):
            flag_read = True
        elif line.startswith(_DIVIDERS[-1]):
            break
        elif flag_read and ':' in line:
            line = line.strip()
            key, value = line.split(':', 1)
            metric_dict[key.strip()] = float(value)

    if metric_dict == {}:
        raise RuntimeError(subp_stdout)
    return metric_dict


def gpcc_test(points, colors):
    write_ply_file(points, 'ScanNet/tmp.ply', rgb=colors.astype(np.uint8))
    command_enc = 'bin/tmc3 --mode=0  ' \
                  '--uncompressedDataPath=ScanNet/tmp.ply ' \
                  '--compressedStreamPath=ScanNet/tmpc ' \
                  '--trisoupNodeSizeLog2=0 ' \
                  '--mergeDuplicatedPoints=0 ' \
                  '--neighbourAvailBoundaryLog2=8 ' \
                  '--intra_pred_max_node_size_log2=6 ' \
                  '--positionQuantizationScale=1 ' \
                  '--inferredDirectCodingMode=1 ' \
                  '--maxNumQtBtBeforeOt=4 ' \
                  '--minQtbtSizeLog2=0 ' \
                  '--planarEnabled=1 ' \
                  '--planarModeIdcmUse=0 ' \
                  '--convertPlyColourspace=1 ' \
                  '--transformType=0 --qp=22 ' \
                  '--qpChromaOffset=0 --bitdepth=8 ' \
                  '--attrOffset=0 --attrScale=1 --attribute=color'
    subp_enc = subprocess.run(
        command_enc, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    )
    command_dec = 'bin/tmc3 --mode=1 ' \
                  '--compressedStreamPath=ScanNet/tmpc ' \
                  '--reconstructedDataPath=ScanNet/tmpr.ply --outputBinaryPly=0'
    subp_dec = subprocess.run(
        command_dec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, check=True, text=True
    )
    psnr = mpeg_pc_error('ScanNet/tmp.ply', 'ScanNet/tmpr.ply', 512, color=True)['c[0],PSNRF']
    bpp = log_extractor.extract_enc_log(subp_enc.stdout)['bpp']
    return bpp, psnr
