import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import type { ECharts, EChartsOption } from 'echarts'
import type { ChipData } from '@/lib/api'
import { useChartTheme, type ChartTheme } from '@/lib/theme'

// 获利盘(现价以下) / 套牢盘(现价以上) 柱色
const PROFIT_COLOR = '#C74040'
const LOSS_COLOR = '#2D9B65'

function argminIdx(arr: number[], target: number): number {
  let best = 0
  let bestDist = Infinity
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target)
    if (d < bestDist) {
      bestDist = d
      best = i
    }
  }
  return best
}

function buildOption(data: ChipData, ct: ChartTheme): EChartsOption {
  const { price_grid, distribution, current_price, avg_cost } = data
  const labels = price_grid.map((p) => p.toFixed(3))
  const currentIdx = argminIdx(price_grid, current_price)
  const avgIdx = argminIdx(price_grid, avg_cost)
  const maxDist = Math.max(1e-9, ...distribution)

  const seriesData = distribution.map((d, i) => ({
    value: d,
    itemStyle: { color: price_grid[i] <= current_price ? PROFIT_COLOR : LOSS_COLOR },
  }))

  return {
    animation: false,
    grid: { left: 8, right: 56, top: 16, bottom: 24, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: ct.tooltipBg,
      borderColor: ct.tooltipBorder,
      textStyle: { color: ct.tooltipText, fontSize: 11 },
      formatter: (params: any) => {
        const p = Array.isArray(params) ? params[0] : params
        if (!p || p.dataIndex == null) return ''
        const idx = p.dataIndex
        const price = price_grid[idx]
        const ratio = distribution[idx]
        return `价格 ${price.toFixed(2)}<br/>筹码占比 ${(ratio * 100).toFixed(2)}%`
      },
    },
    xAxis: {
      type: 'value',
      name: '筹码',
      nameTextStyle: { color: ct.text, fontSize: 10 },
      axisLabel: {
        color: ct.text,
        fontSize: 10,
        formatter: (v: number) => `${(v * 100).toFixed(1)}%`,
      },
      splitLine: { lineStyle: { color: ct.grid } },
      max: maxDist * 1.15,
    },
    yAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: ct.text, fontSize: 10, interval: 9 },
      axisLine: { lineStyle: { color: ct.border } },
    },
    series: [
      {
        type: 'bar',
        data: seriesData,
        barWidth: '100%',
        barCategoryGap: 0,
        markLine: {
          symbol: 'none',
          label: { color: ct.textStrong, fontSize: 10 },
          data: [
            {
              name: `现价 ${current_price.toFixed(2)}`,
              yAxis: currentIdx,
              lineStyle: { color: '#F59E0B', width: 1.5, type: 'dashed' },
            },
            {
              name: `均价 ${avg_cost.toFixed(2)}`,
              yAxis: avgIdx,
              lineStyle: { color: '#8B5CF6', width: 1, type: 'dashed' },
            },
          ],
        },
      },
    ],
  }
}

interface Props {
  data: ChipData
  height?: number
}

export function ChipDistributionChart({ data, height = 480 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<ECharts | null>(null)
  const roRef = useRef<ResizeObserver | null>(null)
  const ct = useChartTheme()

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let chart = chartRef.current
    if (!chart) {
      chart = echarts.init(el, undefined, { renderer: 'canvas' })
      chartRef.current = chart
      roRef.current = new ResizeObserver(() => chart!.resize())
      roRef.current.observe(el)
    }
    chart.setOption(buildOption(data, ct), true)
  }, [data, ct])

  useEffect(() => {
    return () => {
      roRef.current?.disconnect()
      chartRef.current?.dispose()
      chartRef.current = null
      roRef.current = null
    }
  }, [])

  const { profit_ratio, concentration_90, avg_cost_deviation, avg_cost, current_price, single_peak_ratio } = data

  return (
    <div className="w-full">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
        <span>获利盘 {(profit_ratio * 100).toFixed(1)}%</span>
        <span>90% 集中度 {concentration_90.toFixed(3)}</span>
        <span>平均成本 {avg_cost.toFixed(2)}</span>
        <span>偏离度 {(avg_cost_deviation * 100).toFixed(1)}%</span>
        <span>单峰强度 {(single_peak_ratio * 100).toFixed(0)}%</span>
        <span>现价 {current_price.toFixed(2)}</span>
      </div>
      <div ref={containerRef} style={{ width: '100%', height }} />
    </div>
  )
}
