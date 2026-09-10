# Enlarge identical source/output pixels without smoothing for a fair comparison.
Add-Type -AssemblyName System.Drawing
foreach ($item in @(
    @('assets/kodim01.png', 'docs/images/gaussian-detail-before.png'),
    @('docs/images/kodim01-gaussian-blur.png', 'docs/images/gaussian-detail-after.png')
)) {
    $source = [System.Drawing.Bitmap]::new((Join-Path (Get-Location) $item[0]))
    $detail = [System.Drawing.Bitmap]::new(480, 320)
    try {
        for ($row = 0; $row -lt 80; $row++) {
            for ($column = 0; $column -lt 120; $column++) {
                $color = $source.GetPixel(200 + $column, 160 + $row)
                for ($dy = 0; $dy -lt 4; $dy++) {
                    for ($dx = 0; $dx -lt 4; $dx++) {
                        $detail.SetPixel($column * 4 + $dx, $row * 4 + $dy, $color)
                    }
                }
            }
        }
        $detail.Save((Join-Path (Get-Location) $item[1]), [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $source.Dispose()
        $detail.Dispose()
    }
}
