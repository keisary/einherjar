import { useState } from 'react'
import { cn } from '@/lib/utils'

interface Column<T> {
  key: keyof T | string
  header: string
  width?: string
  render?: (row: T) => React.ReactNode
  sortable?: boolean
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  className?: string
  rowKey: (row: T) => string
}

export function DataTable<T>({ columns, data, className, rowKey }: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sortedData = [...data].sort((a, b) => {
    if (!sortKey) return 0
    const aVal = (a as Record<string, unknown>)[sortKey]
    const bVal = (b as Record<string, unknown>)[sortKey]
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDir === 'asc' ? aVal - bVal : bVal - aVal
    }
    return 0
  })

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                style={{ width: col.width }}
                className={cn(
                  'text-left',
                  col.sortable && 'cursor-pointer select-none hover:text-textSecondary'
                )}
                onClick={() => col.sortable && handleSort(String(col.key))}
              >
                <span className="flex items-center gap-1">
                  {col.header}
                  {col.sortable && sortKey === col.key && (
                    <span className="text-frost text-[9px]">
                      {sortDir === 'asc' ? '▲' : '▼'}
                    </span>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row) => (
            <tr key={rowKey(row)} className="transition-colors duration-200">
              {columns.map((col) => (
                <td key={String(col.key)}>
                  {col.render
                    ? col.render(row)
                    : String((row as Record<string, unknown>)[col.key as string] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
