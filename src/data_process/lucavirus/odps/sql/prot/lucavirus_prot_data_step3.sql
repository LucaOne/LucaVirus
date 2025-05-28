--odps sql
--********************************************************************--
--author:sanyuan_xhs_test
--create time:2024-08-05 20:00:59
--********************************************************************--
CREATE FUNCTION lucavirus_prot_label_process AS 'lucavirus_prot_label_process_udf.prot_label_process'
USING 'lucavirus_prot_label_process_udf.py,sanyuan_lucavirus_prot_homo_span_level_label.txt,sanyuan_lucavirus_prot_site_span_level_label.txt,sanyuan_lucavirus_prot_domain_span_level_label.txt,sanyuan_lucavirus_prot_taxonomy_seq_level_label.txt,sanyuan_lucavirus_prot_keyword_seq_level_label.txt,biopython-1.80-cp37.zip' -f ;
-- seq_id, taxid, order_bio, keywords, prot_feature_name,  prot_feature_type, start_p, end_p

DROP TABLE IF EXISTS lucavirus.sanyuan_lucavirus_prot_all_label_detail_step1 ;

CREATE TABLE IF NOT EXISTS lucavirus.sanyuan_lucavirus_prot_all_label_detail_step1 AS
SELECT  prot_seq_accession AS seq_id
     ,seq
     ,lucavirus_prot_label_process(
         prot_seq_accession
        ,taxid
        ,order_bio
        ,keywords
        ,prot_feature_name
        ,prot_feature_type
        ,start_p
        ,end_p
    ) AS labels
FROM
    (
        select *
        from
            lucavirus.sanyuan_lucavirus_prot_uniprot_sprot_label_detail
        union ALL
        select *
        from
            lucavirus.sanyuan_lucavirus_prot_uniprot_trembl_label_detail
        union ALL
        select *
        from
            lucavirus.sanyuan_lucavirus_prot_uniref50_label_all_detail
        union ALL
        select *
        from
            lucavirus.sanyuan_lucavirus_prot_colabfold_envdb_label_all_detail
    )
GROUP BY prot_seq_accession, seq
;


-- 对每个蛋白不存在的那部分label使用占位符

SET odps.sql.python.version=cp37;

CREATE FUNCTION prot_lucavirus_prot_label_fill AS 'lucavirus_prot_label_fill_udf.prot_label_fill' USING 'lucavirus_prot_label_fill_udf.py' -f ;

DROP TABLE IF EXISTS lucavirus.sanyuan_lucavirus_prot_all_label_detail_step2 ;

CREATE TABLE IF NOT EXISTS lucavirus.sanyuan_lucavirus_prot_all_label_detail_step2 AS
SELECT  seq_id
     ,seq
     ,prot_lucavirus_prot_label_fill(seq_id, labels) AS labels
FROM    lucavirus.sanyuan_lucavirus_prot_all_label_detail_step1
;
-- 验证span不能超过序列长度，超过的则去掉

SET odps.sql.python.version=cp37;

CREATE FUNCTION lucavirus_prot_span_verify AS 'lucavirus_prot_span_verify_udf.span_verify' USING 'lucavirus_prot_span_verify_udf.py' -f ;

DROP TABLE IF EXISTS lucavirus.sanyuan_lucavirus_prot_all_label_detail_final ;

CREATE TABLE lucavirus.sanyuan_lucavirus_prot_all_label_detail_final AS
SELECT  seq_id as obj_id
       ,"prot" as obj_type
       ,seq as obj_seq
       ,labels as obj_label
FROM    (
            SELECT  seq_id
                   ,seq
                   ,labels
                   ,lucavirus_prot_span_verify(seq_id, seq, labels) AS flag
            FROM   lucavirus.sanyuan_lucavirus_prot_all_label_detail_step2
        ) t1
WHERE   flag IS NULL
;

-- 序列长度
select min(LENGTH(obj_seq)), MAX(LENGTH(obj_seq)), avg(LENGTH(obj_seq))
from lucavirus.sanyuan_lucavirus_prot_all_label_detail_final;