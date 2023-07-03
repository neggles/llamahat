#ifndef _TUNING_H
#define _TUNING_H

struct ExLlamaTuning {
    int  matmul_recons_thd;
    int  fused_mlp_thd;
    int  sdp_thd;
    bool matmul_fused_remap;

    bool rmsnorm_no_half2;
    bool rope_no_half2;
    bool matmul_no_half2;
    bool silu_no_half2;
    bool concurrent_streams;
};

#endif // _TUNING_H
