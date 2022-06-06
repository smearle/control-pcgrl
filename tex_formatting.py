from pdb import set_trace as TT
import re

def newline(t0, t1, align="r"):
    assert align in ["l", "r", "c"]
    return "\\begin{tabular}["+align+"]{@{}"+align+"@{}}" + t0 + "\\\ " + t1 + "\\end{tabular}"

def align(t0, align="r"):
    assert align in ["l", "r", "c"]
    raise NotImplementedError("align() not implemented properly yet")
    return "\\begin{tabular}["+align+"]{@{}"+align+"@{}}" + t0 + "\\\ " + t0 + "\\end{tabular}"

def pandas_to_latex(df_table, latex_file, vertical_bars=False, right_align_first_column=True, header=True, index=False,
                    escape=False, multicolumn=False, **kwargs) -> None:
    """
    Function that augments pandas DataFrame.to_latex() capability.
    :param df_table: dataframe
    :param latex_file: filename to write latex table code to
    :param vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
    :param right_align_first_column: Allows option to turn off right-aligned first column
    :param header: Whether or not to display the header
    :param index: Whether or not to display the index labels
    :param escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
    :param multicolumn: Enable better handling for multi-index column headers - adds midrules
    :param kwargs: additional arguments to pass through to DataFrame.to_latex()
    :return: None
    """
    n = len(df_table.columns) + len(df_table.index[0])

#   if right_align_first_column:
    cols = 'r' + 'r' * (n - 1)
#   else:
#       cols = 'r' * n

    if vertical_bars:
        # Add the vertical lines
        cols = '|' + '|'.join(cols) + '|'

    latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn,
                              **kwargs)
    latex = latex.replace('\\begin{table}', '\\begin{table*}')
    latex = latex.replace('\end{table}', '\end{table*}')

    if vertical_bars:
        # Remove the booktabs rules since they are incompatible with vertical lines
        latex = re.sub(r'\\(top|mid|bottom)rule', r'\\hline', latex)

    # Multicolumn improvements - center level 1 headers and add midrules
    if multicolumn:
        latex = latex.replace(r'{l}', r'{r}')
        latex = latex.replace(r'{c}', r'{r}')

        offset = len(df_table.index[0])
        #       offset = 1
        midrule_str = ''

        # Find horizontal start and end indices of multicols
        # The below was used for the RL cross eval table.
        #        for i, col in enumerate(df_table.columns.levels[0]):
        #            indices = np.nonzero(np.array(df_table.columns.codes[0]) == i)[0]
        #            hstart = 1 + offset + indices[0]
        #            hend = 1 + offset + indices[-1]
        ##           midrule_str += rf'\cmidrule(lr){{{hstart}-{hend}}} '
        #            midrule_str += rf'\cline{{{hstart}-{hend}}}'
        # Here's a hack for the evo cross eval table.
        hstart = 1 + offset
        hend = offset + len(kwargs['columns'])
        midrule_str += rf'\cline{{{hstart}-{hend}}}'

        # Ensure that headers don't get colored by row highlighting
        #       midrule_str += r'\rowcolor{white}'

        latex_lines = latex.splitlines()
        # FIXME: ad-hoc row indices here
        # TODO: get row index automatically then iterate until we get to the end of the multi-cols
        # latex_lines.insert(7, midrule_str)
        # latex_lines.insert(9, midrule_str)

        # detect start of multicol then add horizontal line between multicol levels
        # for i, l in enumerate(latex_lines):
        #     if '\multicolumn' in l:
        #         mc_start = i
        #         break
        # for i in range(len(df_table.columns[0]) - 1):
        #     latex_lines.insert(mc_start + i + 1, midrule_str)
        latex = '\n'.join(latex_lines)

    with open(latex_file, 'w') as f:
        f.write(latex)

