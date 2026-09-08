import type { ChipData } from '@/lib/api'

/** 筹码峰因子摘要条 —— 在筹码开启时显示最新一天的因子数值。 */
export function ChipFactorSummary({ data }: { data: ChipData }) {
  const items = [
    { label: '获利盘', value: `${(data.profit_ratio * 100).toFixed(1)}%` },
    { label: '90%集中度', value: data.concentration_90.toFixed(3) },
    { label: '偏离度', value: `${(data.avg_cost_deviation * 100).toFixed(1)}%` },
    { label: '单峰强度', value: `${(data.single_peak_ratio * 100).toFixed(0)}%` },
    { label: '单峰宽度', value: `${(data.single_peak_width * 100).toFixed(1)}%` },
    { label: '低位', value: data.is_low_position ? '是' : '否' },
  ]
  return (
    <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-1 px-0.5 text-[10px] leading-4">
      {items.map((it) => (
        <span key={it.label}>
          <span className="text-muted/70">{it.label}</span>{' '}
          <span className="font-medium text-secondary">{it.value}</span>
        </span>
      ))}
    </div>
  )
}
