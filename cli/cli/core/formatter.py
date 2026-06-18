import json
import typing as t


class Formatter:
    @staticmethod
    def table(headers: t.List[str], rows: t.List[t.List[str]], align: t.Optional[t.List[str]] = None) -> str:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        lines = [sep]
        header_line = "|"
        for i, h in enumerate(headers):
            header_line += f" {h:<{col_widths[i]}} |"
        lines.append(header_line)
        lines.append(sep)
        for row in rows:
            line = "|"
            for i, cell in enumerate(row):
                if align and i < len(align) and align[i] == ">":
                    line += f" {str(cell):>{col_widths[i]}} |"
                else:
                    line += f" {str(cell):<{col_widths[i]}} |"
            lines.append(line)
        lines.append(sep)
        return "\n".join(lines)

    @staticmethod
    def json(data: t.Any) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def csv(headers: t.List[str], rows: t.List[t.List[str]]) -> str:
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(str(c) for c in row))
        return "\n".join(lines)

    @staticmethod
    def format_output(
        data: t.Any,
        headers: t.List[str],
        rows: t.List[t.List[str]],
        fmt: str = "table",
    ) -> str:
        if fmt == "json":
            return Formatter.json(data)
        elif fmt == "csv":
            return Formatter.csv(headers, rows)
        return Formatter.table(headers, rows)
