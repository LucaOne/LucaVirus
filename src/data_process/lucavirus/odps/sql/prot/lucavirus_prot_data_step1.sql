--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2024-08-05 19:52:02
--********************************************************************--
--

-- swiss-prot
drop table if exists lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail;
create table if not exists lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail as
select a.prot_seq_accession, a.seq, a.taxid, c.order as order_bio, a.keywords, a.prot_feature_name, a.prot_feature_type, a.start_p, a.end_p
from luca_data2.tmp_lucaone_v2_uniprot_sprot_label_detail_v2 a
join lucavirus.sanyuan_virus_taxid b
on a.taxid = b.taxid
left outer join lucavirus.panyuanfei_util_ncbi_taxid2lineage c
on a.taxid = c.taxid;

-- trembl
drop table if exists lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail;
create table if not exists lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail as
select a.prot_seq_accession, a.seq, a.taxid, c.order as order_bio, a.keywords, a.prot_feature_name, a.prot_feature_type, a.start_p, a.end_p
from luca_data2.tmp_lucaone_v2_uniprot_trembl_label_detail_v2 a
join lucavirus.sanyuan_virus_taxid b
on a.taxid = b.taxid
left outer join lucavirus.panyuanfei_util_ncbi_taxid2lineage c
on a.taxid = c.taxid;

-- uniref50
drop table if exists lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail;
create table if not exists lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail as
select a.prot_seq_accession, a.seq, a.taxid, c.order as order_bio, a.keywords, a.prot_feature_name, a.prot_feature_type, a.start_p, a.end_p
from luca_data2.tmp_lucaone_v2_uniref50_label_detail_all_v2 a
join lucavirus.sanyuan_virus_taxid b
on a.taxid = b.taxid
left outer join lucavirus.panyuanfei_util_ncbi_taxid2lineage c
on a.taxid = c.taxid;

-- colab
drop table if exists lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail;
create table if not exists lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail as
select a.prot_seq_accession, a.seq, a.taxid, c.order as order_bio, a.keywords, a.prot_feature_name, a.prot_feature_type, a.start_p, a.end_p
from luca_data2.tmp_lucaone_v2_colabfold_envdb_label_detail_all_v2 a
join lucavirus.sanyuan_virus_taxid b
on a.taxid = b.taxid
left outer join  lucavirus.panyuanfei_util_ncbi_taxid2lineage c
on a.taxid = c.taxid;
