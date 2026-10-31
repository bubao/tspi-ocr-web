/*
 * 泰山派 RK3566 RGA 2D 硬件加速 C 语言辅助库
 * 基于 Rockchip 官方 librga (im2d + DRM DMA-BUF / PRIME Buffer)
 * 适配 Linux 6.1 内核对硬件连续物理内存的要求，支持任意奇偶尺寸的高速硬件缩放
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rga/RgaApi.h>
#include <rga/drmrga.h>
#include <rga/im2d.h>

#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

static int g_rga_initialized = 0;

int rga_helper_init(void) {
    if (g_rga_initialized) return 0;
    int ret = c_RkRgaInit();
    if (ret == 0) {
        g_rga_initialized = 1;
    }
    return ret;
}

void rga_helper_deinit(void) {
    if (g_rga_initialized) {
        c_RkRgaDeInit();
        g_rga_initialized = 0;
    }
}

int rga_helper_is_available(void) {
    if (!g_rga_initialized) {
        if (rga_helper_init() != 0) return 0;
    }
    return 1;
}

/*
 * 使用 RGA 硬件执行 2D 图像缩放 (基于 DRM DMA-BUF PRIME 物理连续显存)
 * 返回 0 表示硬件加速成功；非 0 表示硬件无法处理，调用方应回退 OpenCV
 */
int rga_helper_resize(const unsigned char *src_data, int src_w, int src_h,
                      unsigned char *dst_data, int dst_w, int dst_h,
                      int channels) {
    if (!src_data || !dst_data) return -1;

    // RGA2 硬件对小于 16px 的极小碎片不支持，直接交由 CPU 处理
    if (src_w < 16 || src_h < 16 || dst_w < 16 || dst_h < 16) {
        return -2;
    }

    // RGA2 硬件对大于 2048px 的超大图物理显存开销极大，直接由 CPU OpenCV 处理，防止耗尽内核 CMA 显存池导致 OOM
    if (src_w > 2048 || src_h > 2048 || dst_w > 2048 || dst_h > 2048) {
        return -8;
    }

    // RGA2 硬件单次缩放限制: 缩放倍率必须在 1/16 ~ 16 倍之间 (0.0625 <= scale <= 16.0)
    // 超过 16 倍缩小或放大时硬件无法单次处理，直接交由 CPU OpenCV 高效处理
    if (src_w > dst_w * 16 || src_h > dst_h * 16 || dst_w > src_w * 16 || dst_h > src_h * 16) {
        return -7;
    }

    if (!g_rga_initialized) {
        if (rga_helper_init() != 0) return -3;
    }

    int bpp = channels * 8;
    int rga_format = (channels == 3) ? RK_FORMAT_BGR_888 : RK_FORMAT_RGBA_8888;

    // RGA2 硬件要求宽高步长满足对齐约束 (4 像素 / 2 像素对齐)
    int sw_align = ALIGN(src_w, 4);
    int dw_align = ALIGN(dst_w, 4);
    int sh_align = ALIGN(src_h, 2);
    int dh_align = ALIGN(dst_h, 2);

    bo_t src_bo, dst_bo;
    if (c_RkRgaGetAllocBuffer(&src_bo, sw_align, sh_align, bpp) != 0) return -4;
    if (c_RkRgaGetAllocBuffer(&dst_bo, dw_align, dh_align, bpp) != 0) {
        c_RkRgaFree(&src_bo);
        return -5;
    }

    int src_dma_fd = -1, dst_dma_fd = -1;
    c_RkRgaGetBufferFd(&src_bo, &src_dma_fd);
    c_RkRgaGetBufferFd(&dst_bo, &dst_dma_fd);

    if (src_dma_fd <= 0 || dst_dma_fd <= 0) {
        c_RkRgaFree(&src_bo);
        c_RkRgaFree(&dst_bo);
        return -6;
    }

    c_RkRgaGetMmap(&src_bo);
    c_RkRgaGetMmap(&dst_bo);

    // 逐行按对齐步长拷贝至 DRM 物理连续显存
    unsigned char *src_ptr = (unsigned char *)src_bo.ptr;
    int src_line_bytes = src_w * channels;
    int src_stride_bytes = sw_align * channels;
    for (int y = 0; y < src_h; y++) {
        memcpy(src_ptr + (size_t)y * src_stride_bytes, src_data + (size_t)y * src_line_bytes, src_line_bytes);
    }

    rga_buffer_t src = wrapbuffer_fd_t(src_dma_fd, src_w, src_h, sw_align, sh_align, rga_format);
    rga_buffer_t dst = wrapbuffer_fd_t(dst_dma_fd, dst_w, dst_h, dw_align, dh_align, rga_format);

    // 调用 RGA 硬件芯片执行 2D 硬件缩放
    IM_STATUS status = imresize(src, dst);

    if (status == IM_STATUS_SUCCESS) {
        unsigned char *dst_ptr = (unsigned char *)dst_bo.ptr;
        int dst_line_bytes = dst_w * channels;
        int dst_stride_bytes = dw_align * channels;
        for (int y = 0; y < dst_h; y++) {
            memcpy(dst_data + (size_t)y * dst_line_bytes, dst_ptr + (size_t)y * dst_stride_bytes, dst_line_bytes);
        }
    }

    c_RkRgaUnmap(&src_bo);
    c_RkRgaUnmap(&dst_bo);
    c_RkRgaFree(&src_bo);
    c_RkRgaFree(&dst_bo);

    return (status == IM_STATUS_SUCCESS) ? 0 : 1;
}
