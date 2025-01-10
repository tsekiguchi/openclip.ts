export type ModelName =
  // OpenAI and LAION 400M models
  | 'openai'
  | 'laion400m_e31'
  | 'laion400m_e32'
  
  // LAION 2B models
  | 'laion2b_e16'
  | 'laion2b_s34b_b79k'
  
  // DataComp-XL models
  | 'datacomp_xl_s13b_b90k'
  
  // DataComp-M models
  | 'datacomp_m_s128m_b4k'
  | 'commonpool_m_clip_s128m_b4k'
  | 'commonpool_m_laion_s128m_b4k'
  | 'commonpool_m_image_s128m_b4k'
  | 'commonpool_m_text_s128m_b4k'
  | 'commonpool_m_basic_s128m_b4k'
  | 'commonpool_m_s128m_b4k'
  
  // DataComp-S models
  | 'datacomp_s_s13m_b4k'
  | 'commonpool_s_clip_s13m_b4k'
  | 'commonpool_s_laion_s13m_b4k'
  | 'commonpool_s_image_s13m_b4k'
  | 'commonpool_s_text_s13m_b4k'
  | 'commonpool_s_basic_s13m_b4k'
  | 'commonpool_s_s13m_b4k'
  
  // MetaClip models
  | 'metaclip_400m'
  | 'metaclip_fullcc';