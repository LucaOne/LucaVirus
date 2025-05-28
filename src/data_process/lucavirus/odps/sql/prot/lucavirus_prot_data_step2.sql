--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2024-08-05 19:52:26
--********************************************************************--

-- span:home:label, size: 1659
-- tunnel download lucavirus.sanyuan_lucavirus_prot_homo_span_level_label sanyuan_lucavirus_prot_homo_span_level_label.txt -fd "," -rd "\n";
drop table if exists lucavirus.sanyuan_lucavirus_prot_homo_span_level_label;
create table if not exists lucavirus.sanyuan_lucavirus_prot_homo_span_level_label
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail
        where prot_feature_type = "Homologous_superfamily"
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail
        where prot_feature_type = "Homologous_superfamily"
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail
        where prot_feature_type = "Homologous_superfamily"
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail
        where prot_feature_type = "Homologous_superfamily"
) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;



-- span:site:label, size: 476
-- tunnel download lucavirus.sanyuan_lucavirus_prot_site_span_level_label sanyuan_lucavirus_prot_site_span_level_label.txt -fd "," -rd "\n";
drop table if exists lucavirus.sanyuan_lucavirus_prot_site_span_level_label;
create table if not exists lucavirus.sanyuan_lucavirus_prot_site_span_level_label
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail
        where prot_feature_type = 'Site'
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail
        where prot_feature_type = 'Site'
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail
        where prot_feature_type = "Site"
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail
        where prot_feature_type = "Site"
) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;

-- span:domain:label, size: 3460
-- tunnel download lucavirus.sanyuan_lucavirus_prot_domain_span_level_label sanyuan_lucavirus_prot_domain_span_level_label.txt -fd "," -rd "\n";
drop table if exists lucavirus.sanyuan_lucavirus_prot_domain_span_level_label;
create table if not exists lucavirus.sanyuan_lucavirus_prot_domain_span_level_label
AS
select distinct prot_feature_name as label
from(
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail
        where prot_feature_type = 'Domain'
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail
        where prot_feature_type = 'Domain'
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail
        where prot_feature_type = "Domain"
        union ALL
        select distinct prot_feature_name
        from lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail
        where prot_feature_type = "Domain"
) t
where prot_feature_name is not NULL and length(prot_feature_name) > 0;


-- seq:taxonomy:label, size: 55
-- tunnel download lucavirus.sanyuan_lucavirus_prot_taxonomy_seq_level_label sanyuan_lucavirus_prot_taxonomy_seq_level_label.txt -fd "," -rd "\n";
drop table if exists lucavirus.sanyuan_lucavirus_prot_taxonomy_seq_level_label;
create table if not exists lucavirus.sanyuan_lucavirus_prot_taxonomy_seq_level_label
AS
select distinct order_bio as label
from(
        select distinct order_bio
        from lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail
        union ALL
        select distinct order_bio
        from lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail
        union ALL
        select distinct order_bio
        from lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail
        union ALL
        select distinct order_bio
        from lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail
) t
where order_bio is not NULL and length(order_bio) > 0;



set odps.sql.python.version=cp37;
create function lucavirus_prot_split_2_multi_rows as 'lucavirus_prot_split_2_multi_rows_udf.split_2_multi_rows' using 'lucavirus_prot_split_2_multi_rows_udf.py' -f;

-- seq:keyword:label, size: 603
-- tunnel download lucavirus.sanyuan_lucavirus_prot_keyword_seq_level_label sanyuan_lucavirus_prot_keyword_seq_level_label.txt -fd "," -rd "\n";
drop table if exists lucavirus.sanyuan_lucavirus_prot_keyword_seq_level_label;
create table if not exists lucavirus.sanyuan_lucavirus_prot_keyword_seq_level_label
AS
select distinct keyword as label
from(
        select DISTINCT keyword
        from(
                select lucavirus_prot_split_2_multi_rows(keywords, ";") as keyword
                from lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail
                union ALL
                select lucavirus_prot_split_2_multi_rows(keywords, ";") as keyword
                from lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail
            ) tmp
        where length(keyword) > 0
) t
where keyword is not NULL and length(keyword) > 0;