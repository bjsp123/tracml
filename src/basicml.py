import typing as tp
import tracdap.rt.api as trac

import pandas as pd

import schemas as schemas
from loan_analysis.loanml import infer_truths


class PandaModel(trac.TracModel):
    def define_parameters(self) -> tp.Dict[str, trac.ModelParameter]:

       return {}
    
    def define_inputs(self) -> tp.Dict[str, trac.ModelInputSchema]:
        approved_loans = trac.load_schema(schemas, "loans_schema_kaggle.csv")

        return {"approved_loans_clean": trac.ModelInputSchema(approved_loans)}

    def define_outputs(self) -> tp.Dict[str, trac.ModelOutputSchema]:
        
        loans_results = trac.load_schema(schemas, "loans_schema_kaggle.csv")

        return {"conclusions": trac.ModelOutputSchema(loans_results)}


    def run_model(self, ctx: trac.TracContext):

        ctx.log().info("Running an ML model")


        loansdf = ctx.get_pandas_table("approved_loans_clean")

        conclusions = infer_truths(loansdf)

        #now output the 'cleaned' data but don't do any actual cleaning as such
        ctx.put_pandas_table("conclusions", conclusions)


if __name__ == "__main__":
    import tracdap.rt.launch as launch
    launch.launch_model(PandaModel, "config/basicml.yaml", "config/sys_config.yaml")