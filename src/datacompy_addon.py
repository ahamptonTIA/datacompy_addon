from io import StringIO
import pandas as pd
import pyspark
import datacompy

#----------------------------------------------------------------------

def pyspark_sql_to_pyspark_pandas(df):
    """
    Converts a DataFrame to a pyspark.pandas.frame.DataFrame 
    if it's a pyspark.sql.dataframe.DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame to potentially convert.

    Returns
    -------
    DataFrame
        The converted DataFrame as a pyspark.pandas.frame.DataFrame, 
        or the original DataFrame if it's already of that type.
    """

    if isinstance(df, pyspark.sql.dataframe.DataFrame):
        # Convert to pandas-on-Spark DataFrame
        if spark.version >= "3.3.0":
            # Spark 3.3.0 and later (recommended)
            return df.pandas_api()
        else:
            return df.to_pandas_on_spark()
    else:
        # Already a pyspark.pandas.frame.DataFrame or pandas dataframe
        return df
      
#----------------------------------------------------------------------

def to_pandas_dataframe(df):
    """
    Converts the input DataFrame to a regular pandas DataFrame.

    Parameters
    ----------
    df : any DataFrame
        The input DataFrame (pandas, Spark SQL, Spark Pandas, or Spark).

    Returns
    -------
    pandas.DataFrame
        The converted pandas DataFrame.

    Raises
    ------
    ValueError
        If the input DataFrame type is not supported.

    """
    if isinstance(df, pyspark.sql.DataFrame):
        # Spark SQL DataFrame, convert to pandas
        return df.toPandas()
    elif isinstance(df, pyspark.pandas.DataFrame):
        # Spark Pandas DataFrame, convert to pandas
        return df.to_pandas()
    elif hasattr(df, "toPandas"):
        # Check for other Spark DataFrames with a toPandas method
        try:
            return df.toPandas()
        except:
            raise ValueError("Unsupported DataFrame type. Please ensure the input is pandas, Spark SQL, Spark Pandas, or a Spark DataFrame with a toPandas method.")
    else:
        return df

#----------------------------------------------------------------------

def remove_df_matching_vals(df, key_fields):
    """
    Cleans a pyspark.pandas DataFrame by setting both "_base" and "_compare" columns to NaN
    when their corresponding values match.

    Parameters
    ----------
    df : pyspark.pandas.DataFrame
        The pyspark.pandas DataFrame to clean.
    key_fields : list
        A list of column names to exclude from matching for deletion.

    Returns
    -------
    pyspark.pandas.DataFrame
        The cleaned pyspark.pandas DataFrame with matching values in "_base" and "_compare" columns replaced with NaN.
    """

    # List columns ending in "_base" that don't start with any key field
    base_cols = [col for col in df.columns if col.endswith("_base") and not col.startswith(tuple(key_fields))]

    # Iterate through remaining "_base" columns
    for base_col in base_cols:
        # Get the corresponding "_compare" column
        compare_col = base_col.replace("_base", "_compare")

        # Check if the "_compare" column exists
        if compare_col in df.columns:
            # Set matching values in both "_base" and "_compare" to NaN
            df.loc[df[base_col] == df[compare_col], [base_col, compare_col]] = None

    return df

#----------------------------------------------------------------------

def compare_dataframes(base_df, compare_df, join_columns, incl_report_text=True, incl_mismatched=True, proc_as_pandas=True):
    """
    Compares two dataframes and returns a dictionary with comparison results.

    Parameters
    ----------
    base_df : pandas DataFrame or Spark DataFrame
        First (base) dataframe to check.
    compare_df : pandas DataFrame or Spark DataFrame
        Second (compare) dataframe to check.
    join_columns : list
        Column(s) to join dataframes on.
    incl_report_text : bool, optional
        Option to include the datacompy comparison report text, True by default.
    incl_mismatched : bool, optional
        Option to include the mismatched rows in the output dictionary, True by default.
    proc_as_pandas : bool, optional
        Option to convert input DataFrames to pandas for comparison which involves less 
        overhead for small DataFrames and avoids a current bug in datacompy pyspark
        processing, True by default.

    Returns
    -------
    dict
        Dictionary containing the comparison results.

    Raises
    ------
    ValueError
        If join_columns is None or empty.
        If base_df and compare_df are not the same type of dataframe.
    """

    def _format_column_stats(column_stats_list):
        """
        Formats the column statistics list.

        Parameters
        ----------
        column_stats_list : list
            List of column statistics dictionaries.

        Returns
        -------
        list
            Formatted column statistics list.
        """
        keep_column_stats = {
            'column': 'column',
            'match_cnt': 'joined_match_cnt',
            'unequal_cnt': 'joined_unequal_cnt',
            'dtype1': 'base_type',
            'dtype2': 'compare_type',
            'all_match': 'all_match'
        }

        column_stats_list = [
            {keep_column_stats[k]: v for k, v in c.items() if k in keep_column_stats.keys()}
            for c in column_stats_list
        ]

        return column_stats_list

    req_max_rows=0 
    init_max_rows=0

    if join_columns is None or join_columns == []:
        raise ValueError("Join_columns cannot be None")

    comp_dict = {}

    if type(base_df) != type(compare_df):
        raise ValueError("base_df and compare_df are not the same type of dataframe")

    if not proc_as_pandas and not (isinstance(base_df, pd.DataFrame) and isinstance(compare_df, pd.DataFrame)):
        # Convert Spark DataFrame to pandas DataFrame
        if isinstance(base_df, pyspark.sql.DataFrame):
            base_df = pyspark_sql_to_pyspark_pandas(base_df)
        if isinstance(compare_df, pyspark.sql.DataFrame):
            compare_df = pyspark_sql_to_pyspark_pandas(compare_df)

        # Check if the base_df and compare_df are Spark DataFrames
        if isinstance(base_df, pyspark.pandas.frame.DataFrame) and isinstance(compare_df, pyspark.pandas.frame.DataFrame):

            # Get the max required row count to compare the dataframes
            init_max_rows = pyspark.pandas.config.get_option("compute.max_rows")
            req_max_rows = max([init_max_rows, len(base_df), len(compare_df)])

            if req_max_rows > init_max_rows:
                # Set a higher limit for compute.max_rows 
                pyspark.pandas.config.set_option("compute.max_rows", req_max_rows) 

            compare = datacompy.SparkCompare(
                df1=base_df,
                df2=compare_df,
                join_columns=join_columns,
                df1_name='base',
                df2_name='compare'
            )


    else:
        try:
            base_df = to_pandas_dataframe(base_df)
            compare_df = to_pandas_dataframe(compare_df)
        except:
            pass

        compare = datacompy.Compare(
            df1=base_df,
            df2=compare_df,
            join_columns=join_columns,
            df1_name='base',
            df2_name='compare'
        )
    # Get counts of each DataFrame
    base_row_cnt = len(base_df)
    comp_row_cnt = len(compare_df)

    # generate the mismatched records dataframe
    if incl_mismatched:
        mismatched_df = compare.all_mismatch(ignore_matching_cols=True)
        postfix_dict = {"_df1": "_base", "_df2": "_compare"}

        new_cols = {
            col: col[:-len(s)] + r
            for col in list(mismatched_df.columns)
            for s, r in postfix_dict.items()
            if col.endswith(s)
        }

        if bool(new_cols):
            mismatched_df = mismatched_df.rename(columns=new_cols)

        # Create similar mismatch column output as SparkCompare
        if not isinstance(mismatched_df, pyspark.pandas.frame.DataFrame):
            base_join_df = compare_df[join_columns].copy()
            base_cols = {c: f"{c}_base" for c in base_join_df.columns if c in join_columns}
            if bool(base_cols):
                # Add additional columns with the same data
                for c in join_columns:
                    base_join_df[f"{c}_base"] = base_join_df[c].copy()

                mismatched_df = mismatched_df.merge(base_join_df, on=join_columns, how='left')

            comp_join_df = compare_df[join_columns].copy()
            comp_cols = {c: f"{c}_compare" for c in comp_join_df.columns if c in join_columns}
            if bool(comp_cols):
                # Add additional columns with the same data
                for c in join_columns:
                    comp_join_df[f"{c}_compare"] = comp_join_df[c].copy()

                mismatched_df = mismatched_df.merge(comp_join_df, on=join_columns, how='left')

            mismatched_df = pyspark.pandas.from_pandas(mismatched_df)
            # Order the columns as SparkCompare does
            join_cols_list = []
            compare_cols = list(mismatched_df.columns)
            for c in join_columns:
                base_name = f'{c}_base'
                comp_name = f'{c}_compare'

                if base_name in compare_cols:
                    join_cols_list.append(base_name)
                    if c in compare_cols:
                        compare_cols.remove(c)
                if comp_name in compare_cols:
                    join_cols_list.append(comp_name)
                    if c in compare_cols:
                        compare_cols.remove(c)

            ordered_cols = join_cols_list + [c for c in compare_cols if c not in join_cols_list]
            mismatched_df = mismatched_df[ordered_cols]
            mismatched_df = remove_df_matching_vals(mismatched_df, join_columns)

    comp_dict = {
        'join_columns': compare.join_columns,
        'removed_columns': list(compare.df1_unq_columns()),
        'new_columns': list(compare.df2_unq_columns()),
        'common_columns': list(compare.intersect_columns()),
        'columns_compared': list(compare.intersect_columns() - set(compare.join_columns)),
        'count_matching_rows': compare.count_matching_rows(),
        'count_unmatched_rows': {
            'base': base_row_cnt - compare.count_matching_rows(),
            'compare': comp_row_cnt - compare.count_matching_rows()
        },
        'column_stats': _format_column_stats(compare.column_stats),
    }

    if incl_mismatched:
        # Export a json representing the mismatched_df
        comp_dict['mismatched'] = 'None' if mismatched_df.empty else mismatched_df.to_json()
    if incl_report_text:
        # generate the report text
        rpt_txt = compare.report()
        comp_dict['report_text'] = rpt_txt

    if req_max_rows > init_max_rows:
        # Reset to the initial default compute.max_rows value
        pyspark.pandas.config.set_option("compute.max_rows", init_max_rows) 
    return comp_dict

#----------------------------------------------------------------------
